"""
90-Day Daily Cash-Flow Simulator.
Implements the recursive daily state equation:
    balance[t] = balance[t-1] + confirmed_income[t] - expenses[t]

Tracks:
- Daily balance trajectory over exactly 90 days from request_date
- Minimum balance reached and date of occurrence (drawdown trough)
- Required reserve to maintain minimum_balance_to_keep
- Maximum safe amount to pay today
- Suppresses double-counting between scheduled debits and inferred recurring debits
"""

from dataclasses import dataclass
from datetime import datetime, timedelta
import calendar
from typing import Dict, List, Optional, Set, Tuple
import pandas as pd
import numpy as np

from .income import ConfirmedIncomeForecaster, RecurringIncomeSchedule
from .expenses import ExpenseForecaster, RecurringExpense


@dataclass
class SimulationResult:
    """Diagnostic and quantitative outputs from the 90-day cash flow simulation."""
    user_id: str
    request_date: str
    current_balance: float
    minimum_balance: float
    requested_amount: float
    next_confirmed_income_date: Optional[str]
    projected_minimum_balance_date: str
    projected_minimum_balance: float
    predicted_reserve: float
    amount_safe_to_pay: float
    daily_balances: List[Tuple[str, float]]


class DailyCashFlowSimulator:
    """Executes a daily cash-flow simulation over the 90-day horizon."""

    def __init__(
        self,
        income_forecaster: ConfirmedIncomeForecaster,
        expense_forecaster: ExpenseForecaster,
        variable_spending_stat: str = 'median',  # 'mean', 'median', 'q75', 'q90'
        forecast_horizon_days: int = 90,
    ):
        self.income_forecaster = income_forecaster
        self.expense_forecaster = expense_forecaster
        self.stat_key = variable_spending_stat
        self.horizon_days = forecast_horizon_days

    def simulate(
        self,
        user_id: str,
        user_events: pd.DataFrame,
        current_available_balance: float,
        minimum_balance_to_keep: float,
        requested_amount: float,
        request_date: str,
    ) -> SimulationResult:
        """Run daily balance recursion and return diagnostic results."""
        req_d = datetime.strptime(str(request_date)[:10], "%Y-%m-%d")

        # 1. Pending debits reserved immediately on day 0 per challenge rules
        pending_total, _ = self.expense_forecaster.get_pending_debits_total(user_events, request_date)
        initial_balance = current_available_balance - pending_total

        # 2. Scheduled events (income & debits) on settlement date
        sched_incomes = self.income_forecaster.get_future_scheduled_income(user_events, request_date)
        sched_income_map: Dict[str, float] = {}
        for s_date, amt in sched_incomes:
            sched_income_map[s_date] = sched_income_map.get(s_date, 0.0) + amt

        # Track dates with explicit scheduled salary credits
        sched_salary_dates = set()
        for _, r in user_events[
            (user_events['category'] == 'salary')
            & (user_events['status'] == 'scheduled')
            & (user_events['direction'] == 'credit')
        ].iterrows():
            sched_salary_dates.add(str(r['settlement_date'])[:10])

        sched_debits = self.expense_forecaster.get_future_scheduled_debits(user_events, request_date)
        sched_debit_map: Dict[str, float] = {}
        scheduled_categories_by_month: Set[Tuple[str, int, int]] = set()  # (cat, year, month)
        for _, s_date, amt, cat in sched_debits:
            sched_debit_map[s_date] = sched_debit_map.get(s_date, 0.0) + amt
            sd_dt = datetime.strptime(s_date, "%Y-%m-%d")
            scheduled_categories_by_month.add((cat, sd_dt.year, sd_dt.month))

        # 3. Recurring commitments
        recurring_commitments = self.expense_forecaster.get_recurring_commitments(
            user_id, user_events, request_date
        )

        # 4. Recurring salary schedule
        salary_schedule = self.income_forecaster.get_recurring_salary_schedule(
            user_id, user_events, request_date
        )

        # 5. Variable essential spending daily rate from calendar-week aggregation
        recurring_cats = set(recurring_commitments.keys())
        var_stats = self.expense_forecaster.get_variable_essential_stats(
            user_events, request_date, exclude_categories=recurring_cats
        )
        weekly_stat = var_stats.get(self.stat_key, var_stats.get('median', 0.0))
        daily_var_rate = weekly_stat / 7.0 if weekly_stat > 0 else 0.0

        # State tracking
        current_bal = initial_balance
        min_balance = current_bal
        min_balance_date = req_d.strftime("%Y-%m-%d")
        next_income_date = None
        daily_trajectory: List[Tuple[str, float]] = [(min_balance_date, current_bal)]

        # Day-by-day recursion over horizon
        for day_offset in range(self.horizon_days):
            cur_date = req_d + timedelta(days=day_offset)
            date_str = cur_date.strftime("%Y-%m-%d")

            daily_income = 0.0
            daily_expenses = 0.0

            # Inflow A: Explicit scheduled income on settlement date
            if date_str in sched_income_map:
                daily_income += sched_income_map[date_str]
                if next_income_date is None and day_offset > 0:
                    next_income_date = date_str

            # Inflow B: Recurring salary (if no explicit scheduled salary today)
            if salary_schedule is not None and date_str not in sched_salary_dates:
                is_pay_day = False
                if salary_schedule.cadence == 'monthly' and salary_schedule.day_of_month is not None:
                    max_m_day = calendar.monthrange(cur_date.year, cur_date.month)[1]
                    target_pay_day = min(salary_schedule.day_of_month, max_m_day)
                    if cur_date.day == target_pay_day:
                        is_pay_day = True
                elif salary_schedule.cadence == 'weekly' and cur_date.weekday() == salary_schedule.day_of_week:
                    is_pay_day = True
                elif salary_schedule.cadence == 'biweekly' and salary_schedule.anchor_date:
                    anchor_dt = datetime.strptime(salary_schedule.anchor_date, "%Y-%m-%d")
                    if (cur_date - anchor_dt).days > 0 and (cur_date - anchor_dt).days % 14 == 0:
                        is_pay_day = True
                elif salary_schedule.cadence == 'triweekly' and salary_schedule.anchor_date:
                    anchor_dt = datetime.strptime(salary_schedule.anchor_date, "%Y-%m-%d")
                    if (cur_date - anchor_dt).days > 0 and (cur_date - anchor_dt).days % 21 == 0:
                        is_pay_day = True

                if is_pay_day:
                    daily_income += salary_schedule.amount
                    if next_income_date is None and day_offset > 0:
                        next_income_date = date_str

            # Outflow A: Scheduled debits on settlement date
            if date_str in sched_debit_map:
                daily_expenses += sched_debit_map[date_str]

            # Outflow B: Inferred recurring commitments (anti-double-counting check)
            for cat, rec_exp in recurring_commitments.items():
                # If an explicit scheduled debit already covers this category in this month, skip inferred
                if (cat, cur_date.year, cur_date.month) in scheduled_categories_by_month:
                    continue

                is_expense_day = False
                if rec_exp.cadence == 'monthly' and rec_exp.day_of_month is not None:
                    max_m_day = calendar.monthrange(cur_date.year, cur_date.month)[1]
                    target_exp_day = min(rec_exp.day_of_month, max_m_day)
                    if cur_date.day == target_exp_day:
                        is_expense_day = True
                elif rec_exp.cadence == 'weekly' and cur_date.weekday() == rec_exp.day_of_week:
                    is_expense_day = True
                elif rec_exp.cadence == 'biweekly' and rec_exp.anchor_date:
                    anchor_dt = datetime.strptime(rec_exp.anchor_date, "%Y-%m-%d")
                    if (cur_date - anchor_dt).days > 0 and (cur_date - anchor_dt).days % 14 == 0:
                        is_expense_day = True
                elif rec_exp.cadence == 'triweekly' and rec_exp.anchor_date:
                    anchor_dt = datetime.strptime(rec_exp.anchor_date, "%Y-%m-%d")
                    if (cur_date - anchor_dt).days > 0 and (cur_date - anchor_dt).days % 21 == 0:
                        is_expense_day = True

                if is_expense_day:
                    daily_expenses += rec_exp.amount

            # Outflow C: Variable essential spending daily rate
            daily_expenses += daily_var_rate

            # State equation: balance[t] = balance[t-1] + income[t] - expenses[t]
            current_bal = current_bal + daily_income - daily_expenses
            daily_trajectory.append((date_str, current_bal))

            if current_bal < min_balance:
                min_balance = current_bal
                min_balance_date = date_str

        # Required reserve to protect minimum balance
        predicted_reserve = max(0.0, current_available_balance - min_balance)

        # Amount safe to pay today
        safe_amount = max(
            0.0,
            min(requested_amount, min_balance - minimum_balance_to_keep)
        )

        return SimulationResult(
            user_id=user_id,
            request_date=str(request_date)[:10],
            current_balance=current_available_balance,
            minimum_balance=minimum_balance_to_keep,
            requested_amount=requested_amount,
            next_confirmed_income_date=next_income_date,
            projected_minimum_balance_date=min_balance_date,
            projected_minimum_balance=min_balance,
            predicted_reserve=predicted_reserve,
            amount_safe_to_pay=safe_amount,
            daily_balances=daily_trajectory,
        )
