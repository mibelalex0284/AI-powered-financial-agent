"""
Deep forensic trace for request_20, request_24, and request_25.
Break down initial balance, pending debits, recurring expenses, variable essential spending,
and identify where the difference between Current and GT safe amounts comes from.
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
sim = DailyCashFlowSimulator(inc_f, exp_f, 'median', 90)

for r_id in ['request_20', 'request_24', 'request_25']:
    r = sample_requests[sample_requests['request_id'] == r_id].iloc[0]
    u_id = r['user_id']
    req_d = str(r['request_date'])[:10]
    req_amt = float(r['requested_amount'])
    p = profiles[profiles['user_id'] == u_id].iloc[0]
    curr_b = float(p['current_available_balance'])
    min_b = float(p['minimum_balance_to_keep'])
    home_c = p['home_currency']

    u_ev = norm.get_user_events(u_id, home_c, req_d)
    res = sim.simulate(u_id, u_ev, curr_b, min_b, req_amt, req_d)

    print(f"\n=======================================================")
    print(f"{r_id} | {u_id} ({home_c}) | Req Date: {req_d}")
    print(f"Current Available Balance: {curr_b:,.2f}")
    print(f"Minimum Balance to Keep:   {min_b:,.2f}")
    print(f"Headroom (Current - Min):  {curr_b - min_b:,.2f}")
    print(f"GT Safe Amount:            {float(r['amount_safe_to_pay']):,.2f}")
    print(f"Predicted Safe Amount:     {res.amount_safe_to_pay:,.2f}")
    print(f"Overestimate Error:        {res.amount_safe_to_pay - float(r['amount_safe_to_pay']):,.2f}")
    print(f"Projected Min Date:        {res.projected_minimum_balance_date}")
    print(f"Projected Min Balance:     {res.projected_minimum_balance:,.2f}")

    # Inspect events between req_date and next payday
    sal_sched = inc_f.get_recurring_salary_schedule(u_id, u_ev, req_d)
    print(f"Salary Schedule: {sal_sched}")
    recs = exp_f.get_recurring_commitments(u_id, u_ev, req_d)
    print(f"Detected Recurring Commitments:")
    for cat, rc in recs.items():
        print(f"  {cat}: {rc.amount:,.2f} ({rc.cadence}, day={rc.day_of_month})")

    # What other settled debits exist in historical events for this user?
    hist_debits = u_ev[(u_ev['status']=='settled') & (u_ev['direction']=='debit') & (u_ev['settlement_date'] < req_d)]
    print("\nAll Historical Debit Categories:")
    for cat in hist_debits['category'].unique():
        c = hist_debits[hist_debits['category']==cat]
        print(f"  {cat:<20}: count={len(c)}, med_amt={c['home_amount'].median():,.2f}")
