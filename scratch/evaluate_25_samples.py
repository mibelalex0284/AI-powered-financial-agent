import os
import sys
sys.path.insert(0, '.')
import pandas as pd
import numpy as np

from code.forecasting import (
    EventNormalizer,
    ExchangeRateProvider,
    ImageAmountResolver,
    ConfirmedIncomeForecaster,
    ExpenseForecaster,
)
from code.optimization import DecisionEngine

def evaluate():
    dataset_dir = 'dataset'
    profiles_df = pd.read_csv(os.path.join(dataset_dir, 'financial_profiles.csv'))
    events_df = pd.read_csv(os.path.join(dataset_dir, 'financial_events.csv'))
    rates_df = pd.read_csv(os.path.join(dataset_dir, 'exchange_rates.csv'))
    sample_requests_df = pd.read_csv(os.path.join(dataset_dir, 'sample_requests.csv'))
    options_df = pd.read_csv(os.path.join(dataset_dir, 'request_payment_options.csv'))
    messages_df = pd.read_csv(os.path.join(dataset_dir, 'messages.csv'))
    images_df = pd.read_csv(os.path.join(dataset_dir, 'images.csv'))

    fx_provider = ExchangeRateProvider(rates_df)
    image_resolver = ImageAmountResolver(images_df)
    normalizer = EventNormalizer(events_df, images_df, fx_provider)
    income_forecaster = ConfirmedIncomeForecaster(messages_df)
    expense_forecaster = ExpenseForecaster(messages_df)

    engine = DecisionEngine(
        event_normalizer=normalizer,
        income_forecaster=income_forecaster,
        expense_forecaster=expense_forecaster,
        variable_spending_stat='median',
    )

    profile_map = {str(p['user_id']).strip(): p for _, p in profiles_df.iterrows()}

    exact_safe = 0
    safe_diffs = []
    norm_diffs = []

    status_matches = 0
    method_matches = 0
    plan_matches = 0
    date_matches = 0
    spend_matches = 0

    table_rows = []

    for _, gt in sample_requests_df.iterrows():
        rid = gt['request_id']
        u_id = str(gt['user_id']).strip()
        prof = profile_map[u_id]
        res = engine.evaluate_request(gt, prof, options_df)

        requested_amt = float(gt.get('requested_amount', 1.0))
        if requested_amt <= 0:
            requested_amt = 1.0

        gt_safe = float(gt['amount_safe_to_pay'])
        pred_safe = float(res.amount_safe_to_pay)
        diff = abs(pred_safe - gt_safe)
        safe_diffs.append(diff)
        norm_diffs.append(diff / requested_amt)

        if diff < 0.5:
            exact_safe += 1

        gt_status = str(gt['affordability_status']).strip()
        pred_status = str(res.affordability_status).strip()
        if gt_status == pred_status:
            status_matches += 1

        gt_method = str(gt['recommended_payment_method']).strip()
        pred_method = str(res.recommended_payment_method).strip()
        if gt_method == pred_method:
            method_matches += 1

        gt_plan = str(gt['payment_plan']).strip()
        pred_plan = str(res.payment_plan).strip()
        if gt_plan == pred_plan:
            plan_matches += 1

        gt_date = str(gt['earliest_date_for_full_payment']).strip() if pd.notna(gt['earliest_date_for_full_payment']) else ""
        pred_date = str(res.earliest_date_for_full_payment).strip() if res.earliest_date_for_full_payment else ""
        if gt_date in ('nan', ''):
            gt_date = ""
        if pred_date in ('nan', ''):
            pred_date = ""
        if gt_date == pred_date:
            date_matches += 1

        gt_spend = str(gt['spending_changes_needed']).strip() if pd.notna(gt['spending_changes_needed']) else ""
        pred_spend = str(res.spending_changes_needed).strip() if res.spending_changes_needed else ""
        if gt_spend in ('nan', ''):
            gt_spend = "none"
        if pred_spend in ('nan', ''):
            pred_spend = "none"
        if gt_spend == pred_spend:
            spend_matches += 1

        table_rows.append({
            'request_id': rid,
            'gt_safe': gt_safe,
            'pred_safe': pred_safe,
            'safe_diff': diff,
            'gt_status': gt_status,
            'pred_status': pred_status,
            'gt_method': gt_method,
            'pred_method': pred_method,
            'gt_plan': gt_plan,
            'pred_plan': pred_plan,
            'gt_date': gt_date,
            'pred_date': pred_date,
            'gt_spend': gt_spend,
            'pred_spend': pred_spend,
            'explanation': res.decision_explanation,
        })

    n = len(sample_requests_df)
    mae = np.mean(safe_diffs)
    norm_error = np.mean(norm_diffs)

    print("=" * 80)
    print("25-SAMPLE EVALUATION BENCHMARK METRICS (PRODUCTION CODE)")
    print("=" * 80)
    print(f"Exact safe-amount matches:     {exact_safe} / {n} ({exact_safe/n*100:.1f}%)")
    print(f"Safe amount MAE:               {mae:,.2f}")
    print(f"Normalized safe amount error:  {norm_error*100:.2f}%")
    print(f"Status accuracy:               {status_matches} / {n} ({status_matches/n*100:.1f}%)")
    print(f"Payment method accuracy:       {method_matches} / {n} ({method_matches/n*100:.1f}%)")
    print(f"Payment plan accuracy:         {plan_matches} / {n} ({plan_matches/n*100:.1f}%)")
    print(f"Earliest-date accuracy:        {date_matches} / {n} ({date_matches/n*100:.1f}%)")
    print(f"Spending-change accuracy:      {spend_matches} / {n} ({spend_matches/n*100:.1f}%)")
    print("=" * 80)

    res_df = pd.DataFrame(table_rows)
    print("\nCOMPLETE 25-REQUEST COMPARISON TABLE:")
    print("-" * 135)
    fmt_str = "{:<12} | {:<12} | {:<12} | {:<10} | {:<14} | {:<14} | {:<14} | {:<14} | {:<10}"
    print(fmt_str.format("Request ID", "GT Safe", "Pred Safe", "Diff", "GT Status", "Pred Status", "GT Method", "Pred Method", "Plan Match"))
    print("-" * 135)
    for _, r in res_df.iterrows():
        p_match = "YES" if r['gt_plan'] == r['pred_plan'] else "NO"
        print(fmt_str.format(
            r['request_id'],
            f"{r['gt_safe']:,.2f}",
            f"{r['pred_safe']:,.2f}",
            f"{r['safe_diff']:,.2f}",
            r['gt_status'][:14],
            r['pred_status'][:14],
            r['gt_method'][:14],
            r['pred_method'][:14],
            p_match,
        ))
    print("-" * 135)

    focus_ids = ['request_06', 'request_07', 'request_11', 'request_13', 'request_19', 'request_20', 'request_24', 'request_25']
    print("\nFORENSIC CHECK OF THE 8 PREVIOUS PROBLEM CASES:")
    print("=" * 135)
    for fid in focus_ids:
        frow = res_df[res_df['request_id'] == fid]
        if not frow.empty:
            fr = frow.iloc[0]
            print(f"[{fid}]")
            print(f"  GT Safe: {fr['gt_safe']:,.2f} | Pred Safe: {fr['pred_safe']:,.2f} | Diff: {fr['safe_diff']:,.2f}")
            print(f"  GT Status: {fr['gt_status']} | Pred Status: {fr['pred_status']}")
            print(f"  GT Method: {fr['gt_method']} | Pred Method: {fr['pred_method']}")
            print(f"  GT Date: '{fr['gt_date']}' | Pred Date: '{fr['pred_date']}'")
            print(f"  GT Spend: '{fr['gt_spend']}' | Pred Spend: '{fr['pred_spend']}'")
            print(f"  GT Plan:   {fr['gt_plan']}")
            print(f"  Pred Plan: {fr['pred_plan']}")
            print(f"  Explanation: {fr['explanation']}")
            print()

if __name__ == '__main__':
    evaluate()
