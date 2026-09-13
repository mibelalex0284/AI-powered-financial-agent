"""
Unified Forecasting Strategy Interface.
Defines:
- Base class ForecastingStrategy
- Concrete strategies:
  1. ExplicitEventsStrategy
  2. RecurringFixedStrategy
  3. FixedPlusMeanEssentialStrategy
  4. FixedPlusMedianEssentialStrategy
  5. FixedPlus75thEssentialStrategy
  6. FixedPlus90thEssentialStrategy
  7. TrueDailySimulationStrategy
"""

from abc import ABC, abstractmethod
from datetime import datetime, timedelta
from typing import Dict, Optional, Tuple
import pandas as pd
import numpy as np

from .income import ConfirmedIncomeForecaster
from .expenses import ExpenseForecaster
from .simulator import DailyCashFlowSimulator, SimulationResult


class ForecastingStrategy(ABC):
    """Abstract base strategy for cash-flow forecasting."""

    def __init__(self, name: str):
        self.name = name

    @abstractmethod
    def predict(
        self,
        user_id: str,
        user_events: pd.DataFrame,
        current_available_balance: float,
        minimum_balance_to_keep: float,
        requested_amount: float,
        request_date: str,
    ) -> SimulationResult:
        """Generate a SimulationResult containing reserve and safe amount predictions."""
        pass


class ExplicitEventsStrategy(ForecastingStrategy):
    """Strategy A: Explicit future events only (pending debits + scheduled debits)."""

    def __init__(self, expense_forecaster: ExpenseForecaster):
        super().__init__("explicit_events")
        self.expense_forecaster = expense_forecaster

    def predict(
        self,
        user_id: str,
        user_events: pd.DataFrame,
        current_available_balance: float,
        minimum_balance_to_keep: float,
        requested_amount: float,
        request_date: str,
    ) -> SimulationResult:
        pending_total, _ = self.expense_forecaster.get_pending_debits_total(user_events, request_date)
        sched_debits = self.expense_forecaster.get_future_scheduled_debits(user_events, request_date)
        sched_total = sum(amt for _, _, amt in sched_debits)

        predicted_reserve = pending_total + sched_total
        headroom = current_available_balance - minimum_balance_to_keep
        safe_amount = max(0.0, min(requested_amount, headroom - predicted_reserve))

        return SimulationResult(
            user_id=user_id,
            request_date=str(request_date)[:10],
            current_balance=current_available_balance,
            minimum_balance=minimum_balance_to_keep,
            requested_amount=requested_amount,
            next_confirmed_income_date=None,
            projected_minimum_balance_date=str(request_date)[:10],
            projected_minimum_balance=current_available_balance - predicted_reserve,
            predicted_reserve=predicted_reserve,
            amount_safe_to_pay=safe_amount,
            daily_balances=[],
        )


class HeuristicTroughStrategy(ForecastingStrategy):
    """
    Base class for window-based drawdown strategies from request_date
    until the next confirmed income date.
    """

    def __init__(
        self,
        name: str,
        income_forecaster: ConfirmedIncomeForecaster,
        expense_forecaster: ExpenseForecaster,
        essential_spending_stat: Optional[str] = None, # 'mean', 'median', 'q75', 'q90', or None
    ):
        super().__init__(name)
        self.income_forecaster = income_forecaster
        self.expense_forecaster = expense_forecaster
        self.stat_key = essential_spending_stat

    def predict(
        self,
        user_id: str,
        user_events: pd.DataFrame,
        current_available_balance: float,
        minimum_balance_to_keep: float,
        requested_amount: float,
        request_date: str,
    ) -> SimulationResult:
        req_d = datetime.strptime(str(request_date)[:10], "%Y-%m-%d")

        # 1. Pending debits
        pending_total, _ = self.expense_forecaster.get_pending_debits_total(user_events, request_date)

        # 2. Next confirmed income date
        salary_schedule = self.income_forecaster.get_recurring_salary_schedule(user_id, user_events, request_date)
        sched_incomes = self.income_forecaster.get_future_scheduled_income(user_events, request_date)

        next_income_d = None
        if sched_incomes:
            next_income_d = datetime.strptime(sched_incomes[0][0], "%Y-%m-%d")
        elif salary_schedule is not None:
            sal_day = salary_schedule[0]
            if req_d.day < sal_day:
                next_income_d = req_d.replace(day=sal_day)
            else:
                if req_d.month == 12:
                    next_income_d = req_d.replace(year=req_d.year + 1, month=1, day=sal_day)
                else:
                    next_income_d = req_d.replace(month=req_d.month + 1, day=sal_day)
        else:
            # Full 90-day horizon if no future income
            next_income_d = req_d + timedelta(days=90)

        window_days = max(1, (next_income_d - req_d).days)

        # 3. Scheduled debits in window
        sched_debits = self.expense_forecaster.get_future_scheduled_debits(user_events, request_date)
        sched_total = 0.0
        for _, s_date, amt in sched_debits:
            sd = datetime.strptime(s_date, "%Y-%m-%d")
            if req_d <= sd < next_income_d:
                sched_total += amt

        # 4. Recurring fixed commitments in window
        monthly_commitments = self.expense_forecaster.get_recurring_monthly_commitments(
            user_id, user_events, request_date
        )
        fixed_total = 0.0
        cur = req_d
        while cur < next_income_d:
            for cat, (m_day, m_amt, _) in monthly_commitments.items():
                if cur.day == m_day:
                    fixed_total += m_amt
            cur += timedelta(days=1)

        # 5. Variable essential spending
        var_total = 0.0
        if self.stat_key is not None:
            var_stats = self.expense_forecaster.get_variable_essential_stats(user_events, request_date)
            for cat, stats in var_stats.items():
                weekly_amt = stats.get(self.stat_key, 0.0)
                var_total += weekly_amt * (window_days / 7.0)

        predicted_reserve = pending_total + sched_total + fixed_total + var_total
        headroom = current_available_balance - minimum_balance_to_keep
        safe_amount = max(0.0, min(requested_amount, headroom - predicted_reserve))

        trough_date_str = (next_income_d - timedelta(days=1)).strftime("%Y-%m-%d")

        return SimulationResult(
            user_id=user_id,
            request_date=str(request_date)[:10],
            current_balance=current_available_balance,
            minimum_balance=minimum_balance_to_keep,
            requested_amount=requested_amount,
            next_confirmed_income_date=next_income_d.strftime("%Y-%m-%d"),
            projected_minimum_balance_date=trough_date_str,
            projected_minimum_balance=current_available_balance - predicted_reserve,
            predicted_reserve=predicted_reserve,
            amount_safe_to_pay=safe_amount,
            daily_balances=[],
        )


class RecurringFixedStrategy(HeuristicTroughStrategy):
    """Strategy B: Recurring fixed commitments only."""
    def __init__(self, income_forecaster: ConfirmedIncomeForecaster, expense_forecaster: ExpenseForecaster):
        super().__init__("recurring_fixed", income_forecaster, expense_forecaster, essential_spending_stat=None)


class FixedPlusMeanEssentialStrategy(HeuristicTroughStrategy):
    """Strategy C: Fixed + Mean essential spending."""
    def __init__(self, income_forecaster: ConfirmedIncomeForecaster, expense_forecaster: ExpenseForecaster):
        super().__init__("fixed_plus_mean_essential", income_forecaster, expense_forecaster, essential_spending_stat='mean')


class FixedPlusMedianEssentialStrategy(HeuristicTroughStrategy):
    """Strategy E1: Fixed + Median essential spending."""
    def __init__(self, income_forecaster: ConfirmedIncomeForecaster, expense_forecaster: ExpenseForecaster):
        super().__init__("fixed_plus_median_essential", income_forecaster, expense_forecaster, essential_spending_stat='median')


class FixedPlus75thEssentialStrategy(HeuristicTroughStrategy):
    """Strategy E2: Fixed + 75th percentile essential spending."""
    def __init__(self, income_forecaster: ConfirmedIncomeForecaster, expense_forecaster: ExpenseForecaster):
        super().__init__("fixed_plus_75th_essential", income_forecaster, expense_forecaster, essential_spending_stat='q75')


class FixedPlus90thEssentialStrategy(HeuristicTroughStrategy):
    """Strategy E3: Fixed + 90th percentile essential spending."""
    def __init__(self, income_forecaster: ConfirmedIncomeForecaster, expense_forecaster: ExpenseForecaster):
        super().__init__("fixed_plus_90th_essential", income_forecaster, expense_forecaster, essential_spending_stat='q90')


class TrueDailySimulationStrategy(ForecastingStrategy):
    """Strategy F (Audited): True 90-day recursive daily balance simulation."""

    def __init__(
        self,
        income_forecaster: ConfirmedIncomeForecaster,
        expense_forecaster: ExpenseForecaster,
        variable_spending_quantile: float = 0.90,
        name_suffix: str = "90th",
    ):
        super().__init__(f"true_daily_simulation_{name_suffix}")
        self.simulator = DailyCashFlowSimulator(
            income_forecaster=income_forecaster,
            expense_forecaster=expense_forecaster,
            variable_spending_quantile=variable_spending_quantile,
            forecast_horizon_days=90,
        )

    def predict(
        self,
        user_id: str,
        user_events: pd.DataFrame,
        current_available_balance: float,
        minimum_balance_to_keep: float,
        requested_amount: float,
        request_date: str,
    ) -> SimulationResult:
        return self.simulator.simulate(
            user_id=user_id,
            user_events=user_events,
            current_available_balance=current_available_balance,
            minimum_balance_to_keep=minimum_balance_to_keep,
            requested_amount=requested_amount,
            request_date=request_date,
        )
