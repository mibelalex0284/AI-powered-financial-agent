import sys
import os
sys.path.insert(0, os.path.abspath('.'))
import pandas as pd
import numpy as np
from datetime import datetime, timedelta

from code.forecasting import (
    EventNormalizer,
    ImageAmountResolver,
    ExchangeRateProvider,
    ConfirmedIncomeForecaster,
    ExpenseForecaster,
    DailyCashFlowSimulator,
)

profiles = pd.read_csv('dataset/financial_profiles.csv')
events = pd.read_csv('dataset/financial_events.csv')
requests = pd.read_csv('dataset/sample_requests.csv')
messages = pd.read_csv('dataset/messages.csv')
rates = pd.read_csv('dataset/exchange_rates.csv')
images = pd.read_csv('dataset/images.csv')

fx = ExchangeRateProvider(rates)
norm = EventNormalizer(events, images, fx)

r23 = requests[requests['request_id'] == 'request_23'].iloc[0]
p23 = profiles[profiles['user_id'] == 'user_23'].iloc[0]
u_events = norm.get_user_events('user_23', 'ZAR', r23['request_date'])

inc_f = ConfirmedIncomeForecaster(messages)
exp_f = ExpenseForecaster()

# User 23 current balance & minimum
curr_bal = float(p23['current_available_balance'])
min_bal = float(p23['minimum_balance_to_keep'])
gt_safe = float(r23['amount_safe_to_pay'])

print(f"User 23 Current Balance: {curr_bal:.2f}")
print(f"User 23 Minimum Balance: {min_bal:.2f}")
print(f"User 23 GT Safe Amount: {gt_safe:.2f}")

# Target reserve in GT:
# Since safe = 9152, and requested = 38016:
# projected_min = min_bal + gt_safe = 27000 + 9152 = 36152.00
# target_reserve = curr_bal - projected_min = 51957.90 - 36152.00 = 15805.90
target_reserve = curr_bal - (min_bal + gt_safe)
print(f"Target Reserve (Drawdown): {target_reserve:.2f}")

# Next salary date for user_23:
# Historical salary settled on 15th of each month (event_1994, etc.)
# Request date is 2025-05-07.
# Next salary is 2025-05-15 (8 days away).
# The trough occurs on 2025-05-14 (day before salary).
# How many days from 2025-05-07 to 2025-05-14? Exactly 7 days!
print("\nDrawdown window: 2025-05-07 to 2025-05-14 (7 days)")

# What are the recurring expenses in this 7-day window?
recs = exp_f.get_recurring_commitments('user_23', u_events, r23['request_date'])
print("\nRecurring commitments for user_23:")
for cat, rc in recs.items():
    print(f"  {cat}: {rc.amount:.2f}, cadence={rc.cadence}, dom={rc.day_of_month}, dow={rc.day_of_week}")

# Check which ones fall between May 7 and May 14
due_recs = []
for cat, rc in recs.items():
    if rc.cadence == 'monthly' and rc.day_of_month is not None:
        if 7 <= rc.day_of_month <= 14:
            due_recs.append((cat, rc.amount, rc.day_of_month))
print("\nCommitments due between May 7 and May 14:")
for cat, amt, dom in due_recs:
    print(f"  {cat} on day {dom}: {amt:.2f}")
total_due_recs = sum(x[1] for x in due_recs)
print(f"Total recurring due before salary: {total_due_recs:.2f}")

# Now, essential spending stats:
var_stats = exp_f.get_variable_essential_stats(u_events, r23['request_date'])
print("\nWeekly essential spending stats for user_23:")
for k, v in var_stats.items():
    print(f"  {k}: {v}")

# Compare predicted reserve for each stat:
for stat_name in ['mean', 'median', 'q75', 'q90']:
    daily_rate = var_stats[stat_name] / 7.0
    var_7d = daily_rate * 7.0 # exactly 1 week!
    total_res = total_due_recs + var_7d
    pred_safe = curr_bal - total_res - min_bal
    diff = pred_safe - gt_safe
    print(f"Stat={stat_name:6s}: weekly_stat={var_stats[stat_name]:.2f}, total_res={total_res:.2f}, pred_safe={pred_safe:.2f}, diff_from_gt={diff:+.2f}")
