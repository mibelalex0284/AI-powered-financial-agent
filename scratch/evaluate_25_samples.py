"""
Benchmark and evaluate the decision pipeline against the 25 solved public examples.
Computes:
- Per-sample comparison table
- Exact safe-amount match count
- MAE, RMSE, max error, normalized error
- Status accuracy, payment method accuracy, payment plan accuracy, earliest date accuracy
"""

import os
import sys
import numpy as np
import pandas as pd

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
from code.optimization import DecisionEngine


def evaluate_samples():
    dataset_dir = 'dataset'
    profiles_df = pd.read_csv(os.path.join(dataset_dir, 'financial_profiles.csv'))
    events_df = pd.read_csv(os.path.join(dataset_dir, 'financial_events.csv'))
    rates_df = pd.read_csv(os.path.join(dataset_dir, 'exchange_rates.csv'))
    sample_df = pd.read_csv(os.path.join(dataset_dir, 'sample_requests.csv'))
    options_df = pd.read_csv(os.path.join(dataset_dir, 'request_payment_options.csv'))
    messages_df = pd.read_csv(os.path.join(dataset_dir, 'messages.csv'))
    images_df = pd.read_csv(os.path.join(dataset_dir, 'images.csv'))

    fx_provider = ExchangeRateProvider(rates_df)
    image_resolver = ImageAmountResolver(images_df)
    normalizer = EventNormalizer(events_df, images_df, fx_provider)
    income_forecaster = ConfirmedIncomeForecaster(messages_df)
    expense_forecaster = ExpenseForecaster()

    engine = DecisionEngine(
        event_normalizer=normalizer,
        income_forecaster=income_forecaster,
        expense_forecaster=expense_forecaster,
        variable_spending_stat='median',
    )

    profile_map = {str(p['user_id']).strip(): p for _, p in profiles_df.iterrows()}

    records = []
    abs_errors = []
    norm_errors = []
    sq_errors = []

    status_matches = 0
    method_matches = 0
    plan_matches = 0
    earliest_matches = 0
    exact_safe_matches = 0

    print("=" * 110)
    print(f"{'Req ID':<10} | {'Pred Safe':<12} | {'GT Safe':<12} | {'Abs Err':<10} | {'Status (P/GT)':<26} | {'Method (P/GT)':<24} | {'Earliest (P/GT)':<22}")
    print("-" * 110)

    for _, row in sample_df.iterrows():
        r_id = str(row['request_id']).strip()
        u_id = str(row['user_id']).strip()
        user_profile = profile_map[u_id]

        res = engine.evaluate_request(
            request_row=row,
            user_profile=user_profile,
            payment_options_df=options_df
        )

        gt_safe = float(row['amount_safe_to_pay'])
        pred_safe = float(res.amount_safe_to_pay)
        abs_err = abs(pred_safe - gt_safe)
        norm_err = abs_err / float(row['requested_amount']) if float(row['requested_amount']) > 0 else 0.0

        abs_errors.append(abs_err)
        norm_errors.append(norm_err)
        sq_errors.append(abs_err ** 2)

        if abs_err < 0.01:
            exact_safe_matches += 1

        gt_status = str(row['affordability_status']).strip()
        pred_status = res.affordability_status
        is_status_match = (gt_status == pred_status)
        if is_status_match:
            status_matches += 1

        gt_method = str(row['recommended_payment_method']).strip()
        pred_method = res.recommended_payment_method
        is_method_match = (gt_method == pred_method)
        if is_method_match:
            method_matches += 1

        gt_plan = str(row['payment_plan']).strip()
        pred_plan = res.payment_plan
        is_plan_match = (gt_plan == pred_plan)
        if is_plan_match:
            plan_matches += 1

        gt_earliest = str(row['earliest_date_for_full_payment']).strip() if pd.notna(row['earliest_date_for_full_payment']) else ""
        pred_earliest = res.earliest_date_for_full_payment
        is_earliest_match = (gt_earliest == pred_earliest)
        if is_earliest_match:
            earliest_matches += 1

        status_str = f"{pred_status[:12]} / {gt_status[:12]}"
        method_str = f"{pred_method[:11]} / {gt_method[:11]}"
        earliest_str = f"{pred_earliest[:10]} / {gt_earliest[:10]}"

        print(f"{r_id:<10} | {pred_safe:<12.2f} | {gt_safe:<12.2f} | {abs_err:<10.2f} | {status_str:<26} | {method_str:<24} | {earliest_str:<22}")

        records.append({
            'request_id': r_id,
            'pred_safe': pred_safe,
            'gt_safe': gt_safe,
            'abs_err': abs_err,
            'norm_err': norm_err,
            'pred_status': pred_status,
            'gt_status': gt_status,
            'pred_method': pred_method,
            'gt_method': gt_method,
            'pred_plan': pred_plan,
            'gt_plan': gt_plan,
            'pred_earliest': pred_earliest,
            'gt_earliest': gt_earliest,
        })

    n = len(sample_df)
    mae = np.mean(abs_errors)
    rmse = np.sqrt(np.mean(sq_errors))
    max_err = np.max(abs_errors)
    norm_err_avg = np.mean(norm_errors) * 100.0

    print("=" * 110)
    print("ACCURACY AND ERROR SUMMARY METRICS (25 SAMPLE REQUESTS):")
    print(f"  Exact Safe Amount Matches: {exact_safe_matches} / {n} ({exact_safe_matches / n * 100:.1f}%)")
    print(f"  MAE (Mean Absolute Error): {mae:,.2f}")
    print(f"  RMSE:                      {rmse:,.2f}")
    print(f"  Max Absolute Error:        {max_err:,.2f}")
    print(f"  Normalized Error (Avg):    {norm_err_avg:.2f}%")
    print(f"  Status Accuracy:           {status_matches} / {n} ({status_matches / n * 100:.1f}%)")
    print(f"  Payment Method Accuracy:   {method_matches} / {n} ({method_matches / n * 100:.1f}%)")
    print(f"  Payment Plan Match Rate:   {plan_matches} / {n} ({plan_matches / n * 100:.1f}%)")
    print(f"  Earliest Date Match Rate:  {earliest_matches} / {n} ({earliest_matches / n * 100:.1f}%)")
    print("=" * 110)


if __name__ == '__main__':
    evaluate_samples()
