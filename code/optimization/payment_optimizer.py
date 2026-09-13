"""
Payment Plan Simulator and Candidate Evaluator.
Evaluates candidate plans against the 90-day daily balance recursion with optional spending changes.
Highly optimized: pre-computes baseline cash flows once per request.
"""

from dataclasses import dataclass
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Set, Tuple
import pandas as pd

from code.forecasting import ConfirmedIncomeForecaster, ExpenseForecaster, RecurringExpense, RecurringIncomeSchedule
from code.optimization.spending_changes import SpendingChangePlan


@dataclass
class Payment:
    date: str
    amount: float


@dataclass
class CandidatePlan:
    payment_method: str  # 'full_payment', 'partial_payment', 'installments', 'wait', 'not_recommended'
    payment_option_id: str
    payments: List[Payment]
    spending_changes: SpendingChangePlan
    total_payable_amount: float
    start_date: str
    completion_date: str
    number_of_payments: int
    is_safe: bool
    completes_by_deadline: bool
    affordability_status: str
    earliest_date_for_full_payment: Optional[str] = None
    explanation: str = ""

    @property
    def plan_string(self) -> str:
        if not self.payments or self.payment_method == 'not_recommended':
            return "none"
        parts = []
        for p in self.payments:
            amt_str = f"{int(round(p.amount))}" if abs(p.amount - round(p.amount)) < 1e-4 else f"{p.amount:.2f}"
            parts.append(f"{p.date}:{amt_str}")
        return "|".join(parts)


@dataclass
class UserBaselineCashflows:
    """Pre-computed baseline cash-flow parameters for a user request."""
    user_id: str
    request_date: str
    initial_balance: float
    minimum_balance_to_keep: float
    sched_income_map: Dict[str, float]
    sched_debit_map: Dict[str, float]
    scheduled_categories_by_month: Set[Tuple[str, int, int]]
    recurring_commitments: List[RecurringExpense]
    salary_schedule: Optional[RecurringIncomeSchedule]
    daily_var_rate: float


class PlanSafetyEvaluator:
    """Simulates a candidate plan against the 90-day daily balance recursion."""

    def __init__(
        self,
        income_forecaster: ConfirmedIncomeForecaster,
        expense_forecaster: ExpenseForecaster,
        variable_spending_stat: str = 'median',
    ):
        self.income_forecaster = income_forecaster
        self.expense_forecaster = expense_forecaster
        self.stat_key = variable_spending_stat

    def build_baseline_cashflows(
        self,
        user_id: str,
        user_events: pd.DataFrame,
        current_available_balance: float,
        minimum_balance_to_keep: float,
        request_date: str,
    ) -> UserBaselineCashflows:
        """Pre-compute all user financial parameters once per request."""
        # 1. Pending debits reserved immediately on day 0
        pending_total, _ = self.expense_forecaster.get_pending_debits_total(user_events, request_date)
        initial_balance = current_available_balance - pending_total

        # 2. Scheduled events
        sched_incomes = self.income_forecaster.get_future_scheduled_income(user_events, request_date)
        sched_income_map = {s_date: amt for s_date, amt in sched_incomes}

        sched_debits = self.expense_forecaster.get_future_scheduled_debits(user_events, request_date)
        sched_debit_map: Dict[str, float] = {}
        scheduled_categories_by_month: Set[Tuple[str, int, int]] = set()
        for _, s_date, amt, cat in sched_debits:
            sched_debit_map[s_date] = sched_debit_map.get(s_date, 0.0) + amt
            sd_dt = datetime.strptime(s_date, "%Y-%m-%d")
            scheduled_categories_by_month.add((cat, sd_dt.year, sd_dt.month))

        # 3. Recurring commitments
        recs = self.expense_forecaster.get_recurring_commitments(user_id, user_events, request_date)
        recurring_commitments = list(recs.values())

        # 4. Salary schedule
        salary_schedule = self.income_forecaster.get_recurring_salary_schedule(user_id, user_events, request_date)

        # 5. Variable essential spending daily rate
        var_stats = self.expense_forecaster.get_variable_essential_stats(user_events, request_date)
        weekly_stat = var_stats.get(self.stat_key, var_stats.get('median', 0.0))
        daily_var_rate = weekly_stat / 7.0 if weekly_stat > 0 else 0.0

        return UserBaselineCashflows(
            user_id=user_id,
            request_date=request_date,
            initial_balance=initial_balance,
            minimum_balance_to_keep=minimum_balance_to_keep,
            sched_income_map=sched_income_map,
            sched_debit_map=sched_debit_map,
            scheduled_categories_by_month=scheduled_categories_by_month,
            recurring_commitments=recurring_commitments,
            salary_schedule=salary_schedule,
            daily_var_rate=daily_var_rate,
        )

    def is_plan_safe(
        self,
        baseline: UserBaselineCashflows,
        payments: List[Payment],
        spending_changes: Optional[SpendingChangePlan] = None,
        horizon_days: int = 90,
    ) -> Tuple[bool, float, str]:
        """
        Pure numerical simulation of balance[t] over 90 days. Extremely fast.
        """
        req_d = datetime.strptime(str(baseline.request_date)[:10], "%Y-%m-%d")

        # Apply spending changes
        stopped_categories = set()
        reduced_categories: Dict[str, float] = {}
        if spending_changes and spending_changes.changes:
            for sc in spending_changes.changes:
                if sc.action == 'stop':
                    stopped_categories.add(sc.category)
                elif sc.action == 'reduce_to':
                    reduced_categories[sc.category] = sc.new_amount

        # Map proposed payments by date
        payment_map: Dict[str, float] = {}
        for p in payments:
            payment_map[p.date] = payment_map.get(p.date, 0.0) + p.amount

        current_bal = baseline.initial_balance
        min_balance = current_bal
        min_balance_date = req_d.strftime("%Y-%m-%d")

        sched_inc = baseline.sched_income_map
        sched_deb = baseline.sched_debit_map
        sched_cats_month = baseline.scheduled_categories_by_month
        sal = baseline.salary_schedule
        recs = baseline.recurring_commitments
        daily_var = baseline.daily_var_rate

        for day_offset in range(horizon_days):
            cur_date = req_d + timedelta(days=day_offset)
            date_str = cur_date.strftime("%Y-%m-%d")

            daily_income = 0.0
            daily_expenses = 0.0

            # Inflow
            if date_str in sched_inc:
                daily_income += sched_inc[date_str]
            elif sal is not None:
                is_pay = False
                if sal.cadence == 'monthly' and cur_date.day == sal.day_of_month:
                    is_pay = True
                elif sal.cadence == 'weekly' and cur_date.weekday() == sal.day_of_week:
                    is_pay = True
                elif sal.cadence == 'biweekly' and sal.anchor_date:
                    anchor_dt = datetime.strptime(sal.anchor_date, "%Y-%m-%d")
                    if (cur_date - anchor_dt).days > 0 and (cur_date - anchor_dt).days % 14 == 0:
                        is_pay = True
                if is_pay:
                    daily_income += sal.amount

            # Outflow: Scheduled debits
            if date_str in sched_deb:
                daily_expenses += sched_deb[date_str]

            # Outflow: Recurring commitments
            for rc in recs:
                if (rc.category, cur_date.year, cur_date.month) in sched_cats_month:
                    continue
                if rc.category in stopped_categories:
                    continue

                rc_amt = reduced_categories.get(rc.category, rc.amount)

                is_due = False
                if rc.cadence == 'monthly' and cur_date.day == rc.day_of_month:
                    is_due = True
                elif rc.cadence == 'weekly' and cur_date.weekday() == rc.day_of_week:
                    is_due = True
                elif rc.cadence == 'biweekly' and rc.anchor_date:
                    anchor_dt = datetime.strptime(rc.anchor_date, "%Y-%m-%d")
                    if (cur_date - anchor_dt).days > 0 and (cur_date - anchor_dt).days % 14 == 0:
                        is_due = True

                if is_due:
                    daily_expenses += rc_amt

            # Outflow: Variable essential
            daily_expenses += daily_var

            # Outflow: Plan payment
            if date_str in payment_map:
                daily_expenses += payment_map[date_str]

            current_bal = current_bal + daily_income - daily_expenses

            if current_bal < min_balance:
                min_balance = current_bal
                min_balance_date = date_str

        is_safe = (min_balance >= baseline.minimum_balance_to_keep - 1e-4)
        return is_safe, min_balance, min_balance_date
