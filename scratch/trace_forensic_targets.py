"""
Forensic trace script for:
- request_13 day-by-day table
- request_06 missing 17.10 reserve
- request_11 underestimating 599,355 reserve
- request_19 partial payment comparison (is it max safe or payment option?)
- request_07 earliest date 2024-10-23 vs 2024-10-15
- requests 20, 24, 25 reserve component analysis
- Diagnostic table
"""

import sys
import os
sys.path.insert(0, os.path.abspath('.'))

import pandas as pd
import numpy as np
from datetime import datetime, timedelta

from code.forecasting import (
    EventNormalizer,
    ExchangeRateProvider,
    ConfirmedIncomeForecaster,
    ExpenseForecaster,
    DailyCashFlowSimulator
)

profiles = pd.read_csv('dataset/financial_profiles.csv')
events = pd.read_csv('dataset/financial_events.csv')
rates = pd.read_csv('dataset/exchange_rates.csv')
sample_requests = pd.read_csv('dataset/sample_requests.csv')
options = pd.read_csv('dataset/request_payment_options.csv')
messages = pd.read_csv('dataset/messages.csv')
images = pd.read_csv('dataset/images.csv')

fx = ExchangeRateProvider(rates)
norm = EventNormalizer(events, images, fx)
inc_f = ConfirmedIncomeForecaster(messages)
exp_f = ExpenseForecaster()

print("=" * 80)
print("PART 1: REQUEST_13 DAY-BY-DAY FORENSIC TRACE")
print("=" * 80)

u13_prof = profiles[profiles['user_id'] == 'user_13'].iloc[0]
u13_events = norm.get_user_events('user_13', 'EUR', '2024-03-07')

# Detailed inspection of request_13
curr_bal = float(u13_prof['current_available_balance'])
min_bal = float(u13_prof['minimum_balance_to_keep'])
req_d_str = '2024-03-07'
req_d = datetime.strptime(req_d_str, "%Y-%m-%d")

# Let's inspect all events of user 13 around this period
pending_total, _ = exp_f.get_pending_debits_total(u13_events, req_d_str)
sched_incomes = inc_f.get_future_scheduled_income(u13_events, req_d_str)
sched_income_map = {s_date: amt for s_date, amt in sched_incomes}
sched_debits = exp_f.get_future_scheduled_debits(u13_events, req_d_str)
sched_debit_map = {}
for _, s_date, amt, cat in sched_debits:
    sched_debit_map[s_date] = sched_debit_map.get(s_date, 0.0) + amt

recs = exp_f.get_recurring_commitments('user_13', u13_events, req_d_str)
var_stats = exp_f.get_variable_essential_stats(u13_events, req_d_str)
daily_var = var_stats['median'] / 7.0

# Print simulation table
print(f"Initial available balance: {curr_bal:.2f} | Minimum balance to keep: {min_bal:.2f}")
print(f"Daily variable essential rate: {daily_var:.2f} EUR/day")
print("Recurring commitments detected by base ExpenseForecaster:")
for cat, rc in recs.items():
    print(f"  {cat}: {rc.amount:.2f} EUR (cadence={rc.cadence}, day={rc.day_of_month})")

print("\nScheduled events:")
for s_date, amt in sched_income_map.items():
    print(f"  Scheduled income on {s_date}: +{amt:.2f}")
for s_date, amt in sched_debit_map.items():
    print(f"  Scheduled debit on {s_date}: -{amt:.2f}")

print("\nDAY-BY-DAY CASH FLOW (from 2024-03-07 to 2024-05-15):")
print(f"{'Date':<10} | {'Opening':<9} | {'ConfInc':<8} | {'SchedInc':<8} | {'RecExp':<8} | {'VarEss':<7} | {'Closing':<9} | {'MinBal':<7} | {'LowestSoFar':<11}")
print("-" * 95)

cur_b = curr_bal - pending_total
lowest_b = cur_b
target_end_d = datetime(2024, 5, 16)
num_days = (target_end_d - req_d).days

for d in range(num_days):
    cur_dt = req_d + timedelta(days=d)
    d_str = cur_dt.strftime("%Y-%m-%d")
    open_b = cur_b
    conf_inc = 0.0
    sched_inc = 0.0
    rec_exp = 0.0
    var_ess = daily_var

    if d_str in sched_income_map:
        sched_inc += sched_income_map[d_str]
    elif cur_dt.day == 15 and cur_dt.month > 3:  # recurring monthly salary on 15th
        conf_inc += 1343.54

    if d_str in sched_debit_map:
        rec_exp += sched_debit_map[d_str]

    for cat, rc in recs.items():
        if rc.cadence == 'monthly' and cur_dt.day == rc.day_of_month:
            rec_exp += rc.amount

    closing_b = open_b + conf_inc + sched_inc - rec_exp - var_ess
    cur_b = closing_b
    if cur_b < lowest_b:
        lowest_b = cur_b

    # Print interesting days
    if d < 10 or cur_dt.day in [14, 15, 16, 1, 2] or cur_dt >= datetime(2024, 5, 10):
        print(f"{d_str:<10} | {open_b:9.2f} | {conf_inc:8.2f} | {sched_inc:8.2f} | {rec_exp:8.2f} | {var_ess:7.2f} | {closing_b:9.2f} | {min_bal:7.2f} | {lowest_b:11.2f}")

print(f"\nLowest balance before May 15 paycheck: {lowest_b:.2f}")
print(f"Headroom above minimum balance {min_bal:.2f}: {lowest_b - min_bal:.2f}")
print(f"Ground truth safe amount: 433.40 EUR")
print(f"Difference: {(lowest_b - min_bal) - 433.40:.2f} EUR")
