import sys
sys.path.insert(0, '.')
import pandas as pd
from datetime import datetime, timedelta
from scratch.trace_forensic_targets import normalizer, engine, profile_map, options_df
from code.optimization.payment_optimizer import Payment

gt = pd.read_csv('dataset/sample_requests.csv')
r07 = gt[gt['request_id'] == 'request_07'].iloc[0]
uid = 'user_07'
prof = profile_map[uid]
req_d = '2024-09-04'
req_amt = float(r07['requested_amount'])
min_bal = float(prof['minimum_balance_to_keep'])
curr_bal = float(prof['current_available_balance'])
ue = normalizer.get_user_events(uid, 'INR', req_d)

base = engine.evaluator.build_baseline_cashflows(uid, ue, curr_bal, min_bal, req_d)
print(f"User 07: Current Bal = {curr_bal}, Min Bal = {min_bal}")
print("Daily var rate:", base.daily_var_rate)
print("Salary schedule:", base.salary_schedule)
print("Recurring commitments:")
for rc in base.recurring_commitments:
    print(" ", rc)

# Payments for 3 installments
payments = [
    Payment('2024-09-12', 68432.0),
    Payment('2024-10-10', 68432.0),
    Payment('2024-11-07', 68432.0),
]

safe_no_spend, min_b, min_b_d = engine.evaluator.is_plan_safe(base, payments, spending_changes=None)
print(f"\nInstallments WITHOUT spending change: safe={safe_no_spend}, min_bal={min_b:.2f} (deficit={min_b - min_bal:.2f}) on {min_b_d}")

# Trace day by day around min_b_d
req_dt = datetime.strptime(req_d, "%Y-%m-%d")
pay_map = {p.date: p.amount for p in payments}
cur_b = base.initial_balance
for d in range(90):
    c_dt = req_dt + timedelta(days=d)
    d_str = c_dt.strftime("%Y-%m-%d")
    inc = 0.0
    exp = 0.0
    if d_str in base.sched_income_map:
        inc += base.sched_income_map[d_str]
    if base.salary_schedule and c_dt.day == base.salary_schedule.day_of_month and d_str not in base.sched_salary_dates:
        inc += base.salary_schedule.amount
    if d_str in base.sched_debit_map:
        exp += base.sched_debit_map[d_str]
    for rc in base.recurring_commitments:
        if rc.cadence == 'monthly' and c_dt.day == rc.day_of_month:
            exp += rc.amount
        elif rc.cadence == 'weekly' and c_dt.weekday() == rc.day_of_week:
            exp += rc.amount
        elif rc.cadence == 'biweekly' and rc.anchor_date:
            anc = datetime.strptime(rc.anchor_date, "%Y-%m-%d")
            if (c_dt - anc).days > 0 and (c_dt - anc).days % 14 == 0:
                exp += rc.amount
        elif rc.cadence == 'triweekly' and rc.anchor_date:
            anc = datetime.strptime(rc.anchor_date, "%Y-%m-%d")
            if (c_dt - anc).days > 0 and (c_dt - anc).days % 21 == 0:
                exp += rc.amount
    exp += base.daily_var_rate
    if d_str in pay_map:
        exp += pay_map[d_str]
    cur_b = cur_b + inc - exp
    if cur_b < min_bal:
        print(f"  DEFICIT on {d_str} (day {d}): Bal = {cur_b:.2f} (diff = {cur_b - min_bal:.2f}) [inc={inc}, exp={exp}]")
