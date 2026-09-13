import sys
sys.path.insert(0, '.')
import os
import pandas as pd
from code.forecasting import (
    EventNormalizer,
    ExchangeRateProvider,
    ImageAmountResolver,
    ConfirmedIncomeForecaster,
    ExpenseForecaster,
)
from code.optimization import DecisionEngine
from code.optimization.payment_optimizer import Payment

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
income_fc = ConfirmedIncomeForecaster(messages_df)
expense_fc = ExpenseForecaster(messages_df)

engine = DecisionEngine(
    event_normalizer=normalizer,
    income_forecaster=income_fc,
    expense_forecaster=expense_fc,
    variable_spending_stat='median',
)

profile_map = {str(p['user_id']).strip(): p for _, p in profiles_df.iterrows()}

def trace_request(req_id):
    gt = sample_requests_df[sample_requests_df['request_id'] == req_id].iloc[0]
    uid = str(gt['user_id']).strip()
    prof = profile_map[uid]
    req_d = str(gt['request_date'])[:10]
    req_amt = float(gt['requested_amount'])
    curr_bal = float(prof['current_available_balance'])
    min_bal = float(prof['minimum_balance_to_keep'])
    home_curr = str(prof['home_currency']).strip()

    ue = normalizer.get_user_events(uid, home_curr, req_d)
    base = engine.evaluator.build_baseline_cashflows(uid, ue, curr_bal, min_bal, req_d)
    sim = engine.simulator.simulate(uid, ue, curr_bal, min_bal, req_amt, req_d)

    print(f"\n==================================================")
    print(f"TRACE FOR {req_id} (user {uid})")
    print(f"Requested: {req_amt} {home_curr} on {req_d}")
    print(f"Current Bal: {curr_bal} | Min Bal: {min_bal}")
    print(f"Sim amount_safe_to_pay today: {sim.amount_safe_to_pay} (GT: {gt['amount_safe_to_pay']})")
    print(f"Sim min_balance in 90d: {sim.minimum_balance} on {sim.projected_minimum_balance_date}")
    print(f"Daily var rate: {base.daily_var_rate}")
    print(f"Salary: {base.salary_schedule}")
    print(f"Recurring commitments: {base.recurring_commitments}")

    sp_plans = engine.spending_optimizer.get_candidate_plans(uid, prof, ue, req_d)
    print(f"Spending plans count: {len(sp_plans)}")
    for i, sp in enumerate(sp_plans):
        print(f"  Plan {i}: {sp.spec_string} (count={sp.count}, savings={sp.total_monthly_savings})")
        # Test full payment today with this sp
        safe_full, min_b, min_b_d = engine.evaluator.is_plan_safe(base, [Payment(req_d, req_amt)], spending_changes=sp)
        print(f"    -> Full payment safe: {safe_full}, min_bal={min_b:.2f} on {min_b_d}")

    # Evaluate complete decision
    res = engine.evaluate_request(gt, prof, options_df)
    print(f"Decision: status={res.affordability_status}, method={res.recommended_payment_method}")
    print(f"  earliest_date={res.earliest_date_for_full_payment}")
    print(f"  spending_changes={res.spending_changes_needed}")
    print(f"  plan={res.payment_plan}")

for r in ['request_06', 'request_11', 'request_07', 'request_13', 'request_21']:
    trace_request(r)
