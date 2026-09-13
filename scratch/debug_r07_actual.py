import sys
sys.path.insert(0, 'd:/hackerrank-orchestrate-september26-main/hackerrank-orchestrate-september26-main/code')
import os
os.chdir('d:/hackerrank-orchestrate-september26-main/hackerrank-orchestrate-september26-main')
import pandas as pd
from data_loader import DataLoader
from forecasting.baseline import BaselineBuilder
from forecasting.expenses import ExpenseForecaster
from optimization.payment_optimizer import PlanSafetyEvaluator

loader = DataLoader()
user_id = 'user_07'
request_id = 'request_07'
req_df = pd.read_csv('dataset/sample_requests.csv')
req = req_df[req_df['request_id']==request_id].iloc[0]
request_date = req['request_date']

profile = loader.load_profile(user_id)
events = loader.load_events(user_id)
fx = loader.load_fx_rates()

builder = BaselineBuilder(profile, events, fx)
base = builder.build(request_date, request_id)

print('Initial balance:', base.initial_balance)
print('Min balance:', base.minimum_balance)
print('Salary schedule:', base.salary_schedule)
print('Recurring commitments:')
for cat, rec in base.recurring_commitments.items():
    print(f'  {cat}: cadence={rec.cadence}, amount={rec.amount:.2f}, anchor={rec.anchor_date}, flex={rec.flexibility}')
print('Daily var rate:', base.daily_variable_rate)

forecaster = ExpenseForecaster(events, profile, request_date)
stats = forecaster.get_variable_essential_stats()
print('Variable stats:', stats)

# Now run the evaluator baseline (no payment)
evaluator = PlanSafetyEvaluator()
safe, min_b, min_d = evaluator.is_plan_safe(base, [], horizon_days=90)
print('Baseline (no payment): safe=' + str(safe) + ', min_bal=' + str(round(min_b,2)) + ' on ' + str(min_d))
print('Safe today (headroom):', round(min_b - base.minimum_balance, 2))
