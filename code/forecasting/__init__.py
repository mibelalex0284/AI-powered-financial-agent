"""
Forecasting and Simulation Package for HackerRank Orchestrate 'Buy or Wait?'.
Provides modular components for:
- Event normalization and status resolution
- Confirmed income forecasting
- Recurring fixed-expense forecasting
- Variable essential-spending forecasting
- 90-day daily cash-flow simulation
- Unified forecasting strategy interface
- Comprehensive regression evaluator
"""

from .normalization import EventNormalizer, ExchangeRateProvider
from .income import ConfirmedIncomeForecaster
from .expenses import ExpenseForecaster
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
    "ConfirmedIncomeForecaster",
    "ExpenseForecaster",
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
