import os
import sys
sys.path.insert(0, '.')
import pandas as pd
from datetime import datetime

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

fx = ExchangeRateProvider(rates_df)
norm = EventNormalizer(events_df, images_df, fx)
inc_fc = ConfirmedIncomeForecaster(messages_df)
exp_fc = ExpenseForecaster(messages_df)
engine = DecisionEngine(norm, inc_fc, exp_fc, variable_spending_stat='median')

def inspect_req(req_id):
    gt = sample_requests_df[sample_requests_df['request_id'] == req_id].iloc[0]
    uid = str(gt['user_id']).strip()
    prof = profiles_df[profiles_df['user_id'] == uid].iloc[0]
    req_d = str(gt['request_date'])[:10]
    curr_bal = float(prof['current_available_balance'])
    min_bal = float(prof['minimum_balance_to_keep'])
    home_curr = str(prof['home_currency']).strip()
    
    ue = norm.get_user_events(uid, home_curr, req_d)
    base = engine.evaluator.build_baseline_cashflows(uid, ue, curr_bal, min_bal, req_d)
    sim = engine.simulator.simulate(uid, ue, curr_bal, min_bal, float(gt['requested_amount']), req_d)
    
    print(f"\n==================== {req_id} ({uid}) ====================")
    print(f"Request Date: {req_d}, Requested: {gt['requested_amount']}")
    print(f"Current Bal: {curr_bal}, Min Bal: {min_bal}")
    print(f"GT Safe: {gt['amount_safe_to_pay']}, Sim Safe: {sim.amount_safe_to_pay}")
    print(f"Diff: {abs(float(gt['amount_safe_to_pay']) - sim.amount_safe_to_pay):.2f}")
    print(f"Daily var rate: {base.daily_var_rate:.4f}")
    
    # Check explicit events scheduled in 90d
    print("\nExplicit scheduled/pending events in next 20 days:")
    for d, amt in sorted(base.sched_income_map.items()):
        print(f"  Income {d}: {amt}")
    for d, amt in sorted(base.sched_debit_map.items()):
        print(f"  Debit {d}: {amt}")

    print("\nRecurring commitments:")
    for rc in base.recurring_commitments:
        print(f"  {rc.category}: {rc.cadence} amt={rc.amount} (flex={rc.flexibility}, dom={rc.day_of_month}, dow={rc.day_of_week}, anchor={rc.anchor_date})")

    print("\nFirst 16 daily simulated balances (without payment):")
    for d, b in sim.daily_balances[:16]:
        print(f"  {d}: bal={b:.2f}")

if __name__ == '__main__':
    for r in ['request_06', 'request_11', 'request_07', 'request_13', 'request_21']:
        inspect_req(r)
