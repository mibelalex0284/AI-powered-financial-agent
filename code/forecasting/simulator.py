"""
90-Day Daily Cash-Flow Simulator.
Implements the recursive daily state equation:
    balance[t] = balance[t-1] + confirmed_income[t] - expenses[t]

Tracks:
- Daily balance trajectory over exactly 90 days from request_date
- Minimum balance reached and date of occurrence (drawdown trough)
- Required reserve to maintain minimum_balance_to_keep
- Maximum safe amount to pay today
"""

from dataclasses import dataclass
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple
import numpy as np
import pandas as pd

from .income import ConfirmedIncomeForecaster
from .expenses import ExpenseForecaster


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
        variable_spending_quantile: float = 0.90,
        forecast_horizon_days: int = 90,
    ):
        self.income_forecaster = income_forecaster
        self.expense_forecaster = expense_forecaster
        self.quantile = variable_spending_quantile
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

        # 1. Pending debits reserved immediately on day 0
        pending_total, _ = self.expense_forecaster.get_pending_debits_total(user_events, request_date)
        initial_balance = current_available_balance - pending_total

        # 2. Scheduled events (income & debits) on settlement date
        sched_incomes = self.income_forecaster.get_future_scheduled_income(user_events, request_date)
        sched_income_map: Dict[str, float] = {}
        for s_date, amt in sched_incomes:
            sched_income_map[s_date] = sched_income_map.get(s_date, 0.0) + amt

        sched_debits = self.expense_forecaster.get_future_scheduled_debits(user_events, request_date)
        sched_debit_map: Dict[str, float] = {}
        for _, s_date, amt in sched_debits:
            sched_debit_map[s_date] = sched_debit_map.get(s_date, 0.0) + amt

        # 3. Recurring monthly commitments
        monthly_commitments = self.expense_forecaster.get_recurring_monthly_commitments(
            user_id, user_events, request_date
        )

        # 4. Recurring salary schedule
        salary_schedule = self.income_forecaster.get_recurring_salary_schedule(
            user_id, user_events, request_date
        )

        # 5. Variable essential spending daily rate
        var_stats = self.expense_forecaster.get_variable_essential_stats(user_events, request_date)
        daily_var_rate = 0.0
        for cat, stats in var_stats.items():
            if self.quantile == 0.0:
                weekly_amt = stats['mean']
            elif self.quantile == 0.50:
                weekly_amt = stats['median']
            elif self.quantile == 0.75:
                weekly_amt = stats['q75']
            elif self.quantile == 0.90:
                weekly_amt = stats['q90']
            else:
                weekly_amt = stats['median']
            daily_var_rate += weekly_amt / 7.0

        # State tracking
        current_bal = initial_balance
        min_balance = current_bal
        min_balance_date = req_d.strftime("%Y-%m-%d")
        next_income_date = None
        daily_trajectory: List[Tuple[str, float]] = [(min_balance_date, current_bal)]

        # Day-by-day recursion
        for day_offset in range(self.horizon_days):
            cur_date = req_d + timedelta(days=day_offset)
            date_str = cur_date.strftime("%Y-%m-%d")

            daily_income = 0.0
            daily_expenses = 0.0

            # Inflow A: Explicit scheduled income
            if date_str in sched_income_map:
                daily_income += sched_income_map[date_str]
                if next_income_date is None and day_offset > 0:
                    next_income_date = date_str

            # Inflow B: Regular recurring salary (if no explicit scheduled income today)
            if salary_schedule is not None:
                sal_day, sal_amt = salary_schedule
                if cur_date.day == sal_day and date_str not in sched_income_map:
                    daily_income += sal_amt
                    if next_income_date is None and day_offset > 0:
                        next_income_date = date_str

            # Outflow A: Scheduled debits on settlement date
            if date_str in sched_debit_map:
                daily_expenses += sched_debit_map[date_str]

            # Outflow B: Monthly recurring commitments
            for cat, (m_day, m_amt, _) in monthly_commitments.items():
                if cur_date.day == m_day:
                    daily_expenses += m_amt

            # Outflow C: Daily essential variable spending
            daily_expenses += daily_var_rate

            # State equation
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
