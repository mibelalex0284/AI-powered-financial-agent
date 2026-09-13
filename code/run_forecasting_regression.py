"""
Forecasting Regression Benchmark Runner.
Executes the full forecasting test suite across all modular strategies and generates:
- Comparative summary metrics (MAE, RMSE, Pearson r, Max Error, Normalized Error %)
- Comprehensive 25-request diagnostic tables
- Focused report on key audit cases: request_05, request_10, request_12, request_23
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
    ImageAmountResolver,
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
    images_df = pd.read_csv(os.path.join(data_dir, "images.csv"))
    rates_df = pd.read_csv(os.path.join(data_dir, "exchange_rates.csv"))

    # Initialize modular forecasting layers
    rate_provider = ExchangeRateProvider(rates_df)
    normalizer = EventNormalizer(events_df, images_df=images_df, rate_provider=rate_provider)
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
        TrueDailySimulationStrategy(income_forecaster, expense_forecaster, variable_spending_stat="mean"),
        TrueDailySimulationStrategy(income_forecaster, expense_forecaster, variable_spending_stat="median"),
        TrueDailySimulationStrategy(income_forecaster, expense_forecaster, variable_spending_stat="q75"),
        TrueDailySimulationStrategy(income_forecaster, expense_forecaster, variable_spending_stat="q90"),
    ]

    evaluator = ForecastingRegressionEvaluator(samples_df, profiles_df, normalizer)
    all_diags, summary_df = evaluator.run_suite(strategies)

    print("\n" + "=" * 105)
    print("REMEDIATED FORECASTING REGRESSION BENCHMARK RESULTS (25 SAMPLE REQUESTS)")
    print("=" * 105)
    print(summary_df.to_string(index=False))

    # Print Detailed Diagnostic Table for the True 90-Day Simulation (median essential)
    sim_name = "true_daily_simulation_median"
    diag_df = all_diags[sim_name]

    print("\n" + "=" * 140)
    print(f"DETAILED DIAGNOSTIC AUDIT TABLE ({sim_name})")
    print("=" * 140)
    diag_cols = [
        'request_id', 'user_id', 'request_date', 'current_balance', 'minimum_balance',
        'next_confirmed_income_date', 'projected_minimum_balance_date', 'projected_minimum_balance',
        'target_reserve', 'predicted_reserve', 'error_reserve', 'predicted_safe_amount',
        'ground_truth_safe_amount', 'error_safe_amount'
    ]
    formatted_diag = diag_df[diag_cols].copy()
    print(formatted_diag.to_string(index=False))

    # Print Focused Report on Key Discrepancy Cases
    print("\n" + "=" * 105)
    print("FOCUSED REMEDIATION VERIFICATION ON KEY AUDIT SAMPLES")
    print("=" * 105)
    audit_ids = ['request_05', 'request_10', 'request_12', 'request_23']
    focus_df = diag_df[diag_df['request_id'].isin(audit_ids)][diag_cols].copy()
    print(focus_df.to_string(index=False))

    # Also compare all strategies on the 4 audit requests
    print("\n" + "-" * 105)
    print("COMPARATIVE SAFE AMOUNT PREDICTIONS ON AUDIT REQUESTS ACROSS STRATEGIES:")
    print("-" * 105)
    comp_records = []
    for s in strategies:
        d = all_diags[s.name]
        row_dict = {'strategy': s.name}
        for rid in audit_ids:
            pred = d[d['request_id'] == rid]['predicted_safe_amount'].iloc[0]
            gt = d[d['request_id'] == rid]['ground_truth_safe_amount'].iloc[0]
            row_dict[rid] = f"{pred:.2f} (gt: {gt:.2f})"
        comp_records.append(row_dict)
    print(pd.DataFrame(comp_records).to_string(index=False))

    # Save outputs to scratch directory
    scratch_dir = os.path.join(repo_root, "..", ".scratch_eval")
    os.makedirs(scratch_dir, exist_ok=True)
    summary_df.to_csv(os.path.join(scratch_dir, "remediated_benchmark_summary.csv"), index=False)
    diag_df.to_csv(os.path.join(scratch_dir, "remediated_sim_median_diagnostics.csv"), index=False)


if __name__ == "__main__":
    main()
