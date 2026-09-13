"""
HackerRank Orchestrate (September 2026) — Buy or Wait?
Main Decision Engine Orchestration Pipeline.

Executes the full evaluation pipeline for all requests in dataset/requests.csv:
1. Loads and normalizes all structured profiles, financial events, and evidence.
2. Resolves missing event amounts from linked images in dataset/images.csv.
3. Resolves foreign exchange strictly using dated conversion rates from exchange_rates.csv.
4. Builds user financial state and initial available cash at request date.
5. Forecasts confirmed income and recurring/essential expenses over the 90-day horizon.
6. Evaluates conservative safe-to-pay headroom and earliest safe full-payment date.
7. Evaluates all eligible candidate plans (immediate, installments, partial, wait).
8. Applies compliant spending-change optimizations where permitted.
9. Ranks feasible plans using the exact challenge specification tie-breakers.
10. Generates audit-compliant, grounded natural language explanations.
11. Generates and validates the final submission output.csv.
"""

import argparse
import os
import sys
import time
from typing import List
import pandas as pd

# Ensure repository root is in sys.path for standalone invocation
REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

from code.forecasting import (
    EventNormalizer,
    ExchangeRateProvider,
    ImageAmountResolver,
    ConfirmedIncomeForecaster,
    ExpenseForecaster,
)
from code.optimization import (
    DecisionEngine,
    DecisionResult,
    OutputValidator,
)


def run_pipeline(
    dataset_dir: str = 'dataset',
    output_path: str = 'output.csv',
    variable_spending_stat: str = 'median',
    validate: bool = True,
) -> pd.DataFrame:
    """Execute the end-to-end financial decision pipeline."""
    start_time = time.time()
    print("=" * 70)
    print("HACKERRANK ORCHESTRATE: BUY OR WAIT? — DECISION PIPELINE")
    print("=" * 70)

    # 1. Load Datasets
    print(f"[1/6] Loading datasets from '{dataset_dir}'...")
    profiles_path = os.path.join(dataset_dir, 'financial_profiles.csv')
    events_path = os.path.join(dataset_dir, 'financial_events.csv')
    rates_path = os.path.join(dataset_dir, 'exchange_rates.csv')
    requests_path = os.path.join(dataset_dir, 'requests.csv')
    options_path = os.path.join(dataset_dir, 'request_payment_options.csv')
    messages_path = os.path.join(dataset_dir, 'messages.csv')
    images_path = os.path.join(dataset_dir, 'images.csv')
    media_dir = os.path.join(dataset_dir, 'media', 'images')

    profiles_df = pd.read_csv(profiles_path)
    events_df = pd.read_csv(events_path)
    rates_df = pd.read_csv(rates_path)
    requests_df = pd.read_csv(requests_path)
    options_df = pd.read_csv(options_path)
    messages_df = pd.read_csv(messages_path)
    images_df = pd.read_csv(images_path)

    print(f"      Loaded {len(requests_df)} evaluation requests.")
    print(f"      Loaded {len(profiles_df)} profiles, {len(events_df)} financial events.")
    print(f"      Loaded {len(options_df)} payment options, {len(rates_df)} FX rates.")

    # 2. Initialize Normalizers and Forecasters
    print("[2/6] Initializing modular forecasting and FX normalization layers...")
    fx_provider = ExchangeRateProvider(rates_df)
    image_resolver = ImageAmountResolver(images_df)
    normalizer = EventNormalizer(events_df, images_df, fx_provider)
    income_forecaster = ConfirmedIncomeForecaster(messages_df)
    expense_forecaster = ExpenseForecaster()

    # 3. Initialize Decision Engine
    print(f"[3/6] Initializing decision optimizer (stat='{variable_spending_stat}')...")
    engine = DecisionEngine(
        event_normalizer=normalizer,
        income_forecaster=income_forecaster,
        expense_forecaster=expense_forecaster,
        variable_spending_stat=variable_spending_stat,
    )

    # 4. Process Requests
    print(f"[4/6] Evaluating all {len(requests_df)} requests...")
    results: List[DecisionResult] = []
    
    # Pre-index profiles by user_id
    profile_map = {str(p['user_id']).strip(): p for _, p in profiles_df.iterrows()}

    for i, (_, req_row) in enumerate(requests_df.iterrows()):
        u_id = str(req_row['user_id']).strip()
        user_profile = profile_map.get(u_id)
        if user_profile is None:
            raise ValueError(f"User profile for {u_id} not found in {profiles_path}")

        res = engine.evaluate_request(
            request_row=req_row,
            user_profile=user_profile,
            payment_options_df=options_df,
        )
        results.append(res)

        if (i + 1) % 50 == 0 or (i + 1) == len(requests_df):
            print(f"      Processed {i + 1}/{len(requests_df)} requests...")

    # 5. Format Output DataFrame
    print("[5/6] Formatting results according to the challenge output contract...")
    output_rows = []
    for r in results:
        output_rows.append({
            'request_id': r.request_id,
            'amount_safe_to_pay': r.amount_safe_to_pay,
            'affordability_status': r.affordability_status,
            'recommended_payment_method': r.recommended_payment_method,
            'payment_plan': r.payment_plan,
            'earliest_date_for_full_payment': r.earliest_date_for_full_payment,
            'spending_changes_needed': r.spending_changes_needed,
            'decision_explanation': r.decision_explanation,
        })

    out_df = pd.DataFrame(output_rows)

    # Write root output.csv
    out_df.to_csv(output_path, index=False)
    print(f"      Wrote predictions to '{output_path}'.")

    # Also sync to dataset/output.csv if distinct
    dataset_out_path = os.path.join(dataset_dir, 'output.csv')
    if os.path.abspath(output_path) != os.path.abspath(dataset_out_path):
        out_df.to_csv(dataset_out_path, index=False)
        print(f"      Synced predictions to '{dataset_out_path}'.")

    # 6. Validate Contract Compliance
    if validate:
        print("[6/6] Validating output.csv against contract constraints...")
        is_valid, errors = OutputValidator.validate_file(output_path, requests_df)
        if is_valid:
            print("      [PASSED] output.csv fully satisfies all challenge contract rules!")
        else:
            print(f"      [FAILED] Found {len(errors)} validation errors:")
            for err in errors[:10]:
                print(f"        - {err}")
            if len(errors) > 10:
                print(f"        ... and {len(errors) - 10} more errors.")
            raise ValueError(f"Validation failed with {len(errors)} errors.")

    elapsed = time.time() - start_time
    print("=" * 70)
    print(f"Completed in {elapsed:.2f} seconds ({elapsed / len(requests_df) * 1000:.1f} ms/request).")
    print("=" * 70)
    return out_df


def main():
    parser = argparse.ArgumentParser(description="Buy or Wait? AI Financial Decision Pipeline")
    parser.add_argument('--dataset_dir', default='dataset', help='Path to directory containing dataset files')
    parser.add_argument('--output_file', default='output.csv', help='Path to write output.csv')
    parser.add_argument('--stat', default='median', choices=['median', 'mean', 'p75', 'p90'], help='Variable spending aggregation statistic')
    parser.add_argument('--no_validate', action='store_true', help='Skip contract validation')
    args = parser.parse_args()

    run_pipeline(
        dataset_dir=args.dataset_dir,
        output_path=args.output_file,
        variable_spending_stat=args.stat,
        validate=not args.no_validate,
    )


if __name__ == '__main__':
    main()
