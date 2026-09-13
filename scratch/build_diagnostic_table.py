"""
Build the required diagnostic table for targeted forensic pass.
Columns:
request_id, GT_safe, predicted_safe, error, projected_min_date,
next_confirmed_income_date, fixed_commitments_before_trough,
variable_essential_reserve, other_reserve, likely_root_cause
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

profiles_map = {str(p['user_id']).strip(): p for _, p in profiles.iterrows()}

target_reqs = ['request_06', 'request_11', 'request_13', 'request_19', 'request_07', 'request_20', 'request_24', 'request_25']

rows = []
for r_id in target_reqs:
    r = sample_requests[sample_requests['request_id'] == r_id].iloc[0]
    u_id = r['user_id']
    req_d_str = str(r['request_date'])[:10]
    req_d = datetime.strptime(req_d_str, "%Y-%m-%d")
    req_amt = float(r['requested_amount'])
    p = profiles_map[u_id]
    curr_b = float(p['current_available_balance'])
    min_b = float(p['minimum_balance_to_keep'])
    home_c = p['home_currency']

    u_ev = norm.get_user_events(u_id, home_c, req_d_str)
    res = sim.simulate(u_id, u_ev, curr_b, min_b, req_amt, req_d_str)

    gt_safe = float(r['amount_safe_to_pay'])
    pred_safe = float(res.amount_safe_to_pay)
    err = pred_safe - gt_safe

    # Break down components before trough
    min_dt = datetime.strptime(res.projected_minimum_balance_date, "%Y-%m-%d")
    days_to_trough = max(0, (min_dt - req_d).days)

    var_stats = exp_f.get_variable_essential_stats(u_ev, req_d_str)
    daily_var = var_stats['median'] / 7.0
    var_reserve = daily_var * days_to_trough

    # Fixed commitments before trough
    recs = exp_f.get_recurring_commitments(u_id, u_ev, req_d_str)
    sched_debits = exp_f.get_future_scheduled_debits(u_ev, req_d_str)
    fixed_before_trough = 0.0
    for d_off in range(days_to_trough + 1):
        cur_day = req_d + timedelta(days=d_off)
        for _, s_date, amt, _ in sched_debits:
            if s_date == cur_day.strftime("%Y-%m-%d"):
                fixed_before_trough += amt
        for _, rc in recs.items():
            if rc.cadence == 'monthly' and cur_day.day == rc.day_of_month:
                fixed_before_trough += rc.amount

    other_reserve = (curr_b - min_b) - (fixed_before_trough + var_reserve) - pred_safe

    # Likely root cause
    causes = {
        'request_06': "Marginal headroom: short by 17.10 EUR before next payday, requiring stopping event_476 (streaming)",
        'request_11': "Omitted recurring dining event_989 (Weekend food delivery, 1.16M IDR) reducible to 665k IDR",
        'request_13': "Erroneous biweekly salary cadence from interleaved second income; omitted biweekly dining & monthly entertainment",
        'request_19': "Omitted monthly shopping (5,772 INR); first partial payment is amount_safe_to_pay, not a payment option",
        'request_07': "Ignored message_05 payroll date update to 23rd; credit on 15th caused 8-day early date prediction",
        'request_20': "Omitted recurring dining (3,352 INR/3wks) and entertainment (2,115 INR/mo) before next payday",
        'request_24': "Omitted recurring dining (1,902 INR/wk), shopping (2,514 INR/mo), and entertainment (1,896 INR/mo)",
        'request_25': "Omitted recurring weekly dining (1.05M IDR/wk) and monthly entertainment (451k IDR) before next payday"
    }

    rows.append({
        'request_id': r_id,
        'GT_safe': gt_safe,
        'predicted_safe': pred_safe,
        'error': err,
        'projected_min_date': res.projected_minimum_balance_date,
        'next_confirmed_income_date': res.next_confirmed_income_date,
        'fixed_commitments_before_trough': fixed_before_trough,
        'variable_essential_reserve': var_reserve,
        'other_reserve': other_reserve,
        'likely_root_cause': causes.get(r_id, "")
    })

diag_df = pd.DataFrame(rows)
pd.set_option('display.max_columns', 15)
pd.set_option('display.width', 1000)
print(diag_df.to_string(index=False))
