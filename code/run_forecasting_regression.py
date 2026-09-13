"""
Forecasting Regression Benchmark Runner.
Executes the full forecasting test suite across all modular strategies and generates:
- Comparative summary metrics (MAE, RMSE, Pearson r, Max Error)
- Comprehensive 25-request diagnostic tables
- Forensic comparison against ground-truth sample requests
"""

import os
import sys
import pandas as pd

# Add code directory to sys.path
code_dir = os.path.abspath(os.path.join(os.path.dirname(__file__)))
repo_root = os.path.abspath(os.path.join(code_dir, ".."))
if code_dir not in sys.path:
    sys.path.insert(0, code_dir)
if repo_root not in sys.path:
    sys.path.insert(0, repo_root)

from forecasting import (
    EventNormalizer,
    ExchangeRateProvider,
    ConfirmedIncomeForecaster,
    ExpenseForecaster,
    ExplicitEventsStrategy,
    RecurringFixedStrategy,
    FixedPlusMeanEssentialStrategy,
    FixedPlusMedianEssentialStrategy,
    FixedPlus75thEssentialStrategy,
    FixedPlus90thEssentialStrategy,
    TrueDailySimulationStrategy,
    ForecastingRegressionEvaluator,
)

def main():
    data_dir = os.path.join(repo_root, "dataset")

    # Load data
    samples_df = pd.read_csv(os.path.join(data_dir, "sample_requests.csv"))
    profiles_df = pd.read_csv(os.path.join(data_dir, "financial_profiles.csv"))
    events_df = pd.read_csv(os.path.join(data_dir, "financial_events.csv"))
    messages_df = pd.read_csv(os.path.join(data_dir, "messages.csv"))
    rates_df = pd.read_csv(os.path.join(data_dir, "exchange_rates.csv"))

    # Initialize modular forecasting layers
    rate_provider = ExchangeRateProvider(rates_df)
    normalizer = EventNormalizer(events_df, rate_provider=rate_provider)
    income_forecaster = ConfirmedIncomeForecaster(messages_df)
    expense_forecaster = ExpenseForecaster(messages_df)

    # Instantiate strategies behind the unified interface
    strategies = [
        ExplicitEventsStrategy(expense_forecaster),
        RecurringFixedStrategy(income_forecaster, expense_forecaster),
        FixedPlusMeanEssentialStrategy(income_forecaster, expense_forecaster),
        FixedPlusMedianEssentialStrategy(income_forecaster, expense_forecaster),
        FixedPlus75thEssentialStrategy(income_forecaster, expense_forecaster),
        FixedPlus90thEssentialStrategy(income_forecaster, expense_forecaster),
        TrueDailySimulationStrategy(income_forecaster, expense_forecaster, variable_spending_quantile=0.50, name_suffix="50th"),
        TrueDailySimulationStrategy(income_forecaster, expense_forecaster, variable_spending_quantile=0.75, name_suffix="75th"),
        TrueDailySimulationStrategy(income_forecaster, expense_forecaster, variable_spending_quantile=0.90, name_suffix="90th"),
        TrueDailySimulationStrategy(income_forecaster, expense_forecaster, variable_spending_quantile=0.95, name_suffix="95th"),
    ]

    evaluator = ForecastingRegressionEvaluator(samples_df, profiles_df, normalizer)
    all_diags, summary_df = evaluator.run_suite(strategies)

    print("\n" + "=" * 100)
    print("FORECASTING REGRESSION BENCHMARK RESULTS (25 SAMPLE REQUESTS)")
    print("=" * 100)
    print(summary_df.to_string(index=False))

    # Print Detailed Diagnostic Table for the True 90-Day Simulation (90th percentile)
    best_sim_name = "true_daily_simulation_90th"
    diag_df = all_diags[best_sim_name]

    print("\n" + "=" * 135)
    print(f"DETAILED DIAGNOSTIC AUDIT TABLE ({best_sim_name})")
    print("=" * 135)
    diag_cols = [
        'request_id', 'user_id', 'request_date', 'current_balance', 'minimum_balance',
        'next_confirmed_income_date', 'projected_minimum_balance_date', 'projected_minimum_balance',
        'target_reserve', 'predicted_reserve', 'error_reserve', 'predicted_safe_amount',
        'ground_truth_safe_amount', 'error_safe_amount'
    ]
    formatted_diag = diag_df[diag_cols].copy()
    print(formatted_diag.to_string(index=False))

    # Save outputs to scratch directory
    scratch_dir = os.path.join(repo_root, "..", ".scratch_eval")
    os.makedirs(scratch_dir, exist_ok=True)
    summary_df.to_csv(os.path.join(scratch_dir, "forecasting_benchmark_summary.csv"), index=False)
    diag_df.to_csv(os.path.join(scratch_dir, "true_sim_90th_diagnostics.csv"), index=False)

if __name__ == "__main__":
    main()
