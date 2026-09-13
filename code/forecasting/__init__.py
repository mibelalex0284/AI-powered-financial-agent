"""
Forecasting and Simulation Package for HackerRank Orchestrate 'Buy or Wait?'.
Provides modular components for:
- Event normalization, image amount resolution, and exchange rate resolution
- Confirmed income forecasting
- Recurring fixed-expense forecasting
- Variable essential-spending forecasting
- 90-day daily cash-flow simulation
- Unified forecasting strategy interface
- Comprehensive regression evaluator
"""

from .normalization import EventNormalizer, ExchangeRateProvider, ImageAmountResolver, RateResolution
from .income import ConfirmedIncomeForecaster, RecurringIncomeSchedule
from .expenses import ExpenseForecaster, RecurringExpense
from .simulator import DailyCashFlowSimulator, SimulationResult
from .strategies import (
    ForecastingStrategy,
    ExplicitEventsStrategy,
    RecurringFixedStrategy,
    FixedPlusMeanEssentialStrategy,
    FixedPlusMedianEssentialStrategy,
    FixedPlus75thEssentialStrategy,
    FixedPlus90thEssentialStrategy,
    TrueDailySimulationStrategy,
)
from .evaluator import ForecastingRegressionEvaluator

__all__ = [
    "EventNormalizer",
    "ExchangeRateProvider",
    "ImageAmountResolver",
    "RateResolution",
    "ConfirmedIncomeForecaster",
    "RecurringIncomeSchedule",
    "ExpenseForecaster",
    "RecurringExpense",
    "DailyCashFlowSimulator",
    "SimulationResult",
    "ForecastingStrategy",
    "ExplicitEventsStrategy",
    "RecurringFixedStrategy",
    "FixedPlusMeanEssentialStrategy",
    "FixedPlusMedianEssentialStrategy",
    "FixedPlus75thEssentialStrategy",
    "FixedPlus90thEssentialStrategy",
    "TrueDailySimulationStrategy",
    "ForecastingRegressionEvaluator",
]
