"""
Targeted forensic pass on requests 06, 11, 13, 19, 07, 20, 24, 25.
Inspects raw events, profiles, options, messages, and exact cash flows.
"""

import sys
import os
sys.path.insert(0, os.path.abspath('.'))

import pandas as pd
import numpy as np
from datetime import datetime, timedelta

events = pd.read_csv('dataset/financial_events.csv')
profiles = pd.read_csv('dataset/financial_profiles.csv')
sample_requests = pd.read_csv('dataset/sample_requests.csv')
options = pd.read_csv('dataset/request_payment_options.csv')
messages = pd.read_csv('dataset/messages.csv')
images = pd.read_csv('dataset/images.csv')
rates = pd.read_csv('dataset/exchange_rates.csv')

from code.forecasting import (
    EventNormalizer,
    ExchangeRateProvider,
    ConfirmedIncomeForecaster,
    ExpenseForecaster,
    DailyCashFlowSimulator
)

fx = ExchangeRateProvider(rates)
norm = EventNormalizer(events, images, fx)
inc_f = ConfirmedIncomeForecaster(messages)
exp_f = ExpenseForecaster()

def inspect_user_recurrence(user_id, home_curr, req_date):
    req_d = str(req_date)[:10]
    u_events = norm.get_user_events(user_id, home_curr, req_d)
    hist = u_events[(u_events['status']=='settled') & (u_events['direction']=='debit') & (u_events['settlement_date'] < req_d)]
    
    print(f"\n--- Recurrence check for {user_id} ({home_curr}) before {req_d} ---")
    for cat in hist['category'].unique():
        c_ev = hist[hist['category']==cat].sort_values('settlement_date')
        if len(c_ev) >= 2:
            dates = pd.to_datetime(c_ev['settlement_date'])
            diffs = dates.diff().dt.days.dropna()
            med_diff = diffs.median()
            amts = c_ev['home_amount']
            print(f"  Category '{cat:<20}': count={len(c_ev):2d}, med_interval={med_diff:4.1f} days, med_amt={amts.median():10.2f}, flex={c_ev['flexibility'].iloc[-1]}")

target_reqs = ['request_06', 'request_11', 'request_13', 'request_19', 'request_07', 'request_20', 'request_24', 'request_25']

for r_id in target_reqs:
    r_row = sample_requests[sample_requests['request_id']==r_id].iloc[0]
    u_id = r_row['user_id']
    req_d = r_row['request_date']
    u_prof = profiles[profiles['user_id']==u_id].iloc[0]
    home_curr = u_prof['home_currency']
    print(f"\n==================================================================")
    print(f"REQUEST {r_id} | User {u_id} ({home_curr}) | Req Date {req_d} | Req Amt {r_row['requested_amount']}")
    print(f"GT: safe={r_row['amount_safe_to_pay']}, status={r_row['affordability_status']}, method={r_row['recommended_payment_method']}")
    print(f"GT Plan: {r_row['payment_plan']}, Changes: {r_row['spending_changes_needed']}, Earliest: {r_row['earliest_date_for_full_payment']}")
    inspect_user_recurrence(u_id, home_curr, req_d)
