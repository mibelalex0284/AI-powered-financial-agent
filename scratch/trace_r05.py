import sys
import os
sys.path.insert(0, os.path.abspath('.'))
import pandas as pd
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

r05 = requests[requests['request_id'] == 'request_05'].iloc[0]
p05 = profiles[profiles['user_id'] == 'user_05'].iloc[0]
u_events = norm.get_user_events('user_05', 'ZAR', r05['request_date'])

inc_f = ConfirmedIncomeForecaster(messages)
exp_f = ExpenseForecaster()

req_d = datetime.strptime(str(r05['request_date'])[:10], '%Y-%m-%d')
comp_d = datetime.strptime(str(r05['desired_completion_date'])[:10], '%Y-%m-%d')
horizon_90_d = req_d + timedelta(days=90)

print(f"Request Date: {req_d.strftime('%Y-%m-%d')}")
print(f"Desired Completion Date: {comp_d.strftime('%Y-%m-%d')} ({(comp_d - req_d).days} days)")
print(f"90-day Horizon End: {horizon_90_d.strftime('%Y-%m-%d')}")

# Trace daily simulation manually to output all columns requested
# - opening balance
# - confirmed income
# - scheduled income
# - recurring fixed expenses
# - variable essential spending
# - requested purchase/payment
# - closing balance
# - minimum balance
# - lowest balance/date

pending_total, _ = exp_f.get_pending_debits_total(u_events, r05['request_date'])
initial_balance = float(p05['current_available_balance']) - pending_total

sched_incomes = inc_f.get_future_scheduled_income(u_events, r05['request_date'])
sched_income_map = {s_date: amt for s_date, amt in sched_incomes}

sched_debits = exp_f.get_future_scheduled_debits(u_events, r05['request_date'])
sched_debit_map = {}
scheduled_categories_by_month = set()
for _, s_date, amt, cat in sched_debits:
    sched_debit_map[s_date] = sched_debit_map.get(s_date, 0.0) + amt
    sd_dt = datetime.strptime(s_date, "%Y-%m-%d")
    scheduled_categories_by_month.add((cat, sd_dt.year, sd_dt.month))

recurring_commitments = list(exp_f.get_recurring_commitments('user_05', u_events, r05['request_date']).values())
salary_schedule = inc_f.get_recurring_salary_schedule('user_05', u_events, r05['request_date'])

var_stats = exp_f.get_variable_essential_stats(u_events, r05['request_date'])
weekly_stat = var_stats.get('median', 0.0)
daily_var_rate = weekly_stat / 7.0

rows = []
curr_bal = initial_balance
min_bal_so_far = curr_bal
min_bal_date = req_d.strftime('%Y-%m-%d')

for day_offset in range(90):
    cur_date = req_d + timedelta(days=day_offset)
    date_str = cur_date.strftime('%Y-%m-%d')
    opening_bal = curr_bal

    conf_inc = 0.0
    sched_inc = sched_income_map.get(date_str, 0.0)

    # Recurring fixed expenses
    rec_fixed = 0.0
    # Check scheduled debits first
    sched_deb = sched_debit_map.get(date_str, 0.0)
    rec_fixed += sched_deb

    for rc in recurring_commitments:
        if (rc.category, cur_date.year, cur_date.month) in scheduled_categories_by_month:
            continue
        is_due = False
        if rc.cadence == 'monthly' and cur_date.day == rc.day_of_month:
            is_due = True
        elif rc.cadence == 'weekly' and cur_date.weekday() == rc.day_of_week:
            is_due = True
        elif rc.cadence == 'biweekly' and rc.anchor_date:
            anchor_dt = datetime.strptime(rc.anchor_date, "%Y-%m-%d")
            if (cur_date - anchor_dt).days > 0 and (cur_date - anchor_dt).days % 14 == 0:
                is_due = True
        if is_due:
            rec_fixed += rc.amount

    var_spend = daily_var_rate
    req_payment = 0.0  # Safe amount analysis checks balance before payment

    closing_bal = opening_bal + conf_inc + sched_inc - rec_fixed - var_spend
    curr_bal = closing_bal

    if closing_bal < min_bal_so_far:
        min_bal_so_far = closing_bal
        min_bal_date = date_str

    rows.append({
        'day': day_offset,
        'date': date_str,
        'opening_balance': opening_bal,
        'confirmed_income': conf_inc,
        'scheduled_income': sched_inc,
        'recurring_fixed_expenses': rec_fixed,
        'variable_essential_spending': var_spend,
        'requested_purchase_payment': req_payment,
        'closing_balance': closing_bal,
        'minimum_balance_to_keep': float(p05['minimum_balance_to_keep']),
        'lowest_balance_so_far': min_bal_so_far,
        'lowest_date_so_far': min_bal_date,
    })

df = pd.DataFrame(rows)
df.to_csv('scratch/r05_trace.csv', index=False)

# Summary at desired completion date (2026-01-12, day 67)
df_comp = df[df['date'] <= '2026-01-12']
min_comp = df_comp['closing_balance'].min()
min_comp_row = df_comp.loc[df_comp['closing_balance'].idxmin()]

print(f"\n--- AT DESIRED COMPLETION DATE (2026-01-12, day {min_comp_row['day']}) ---")
print(f"Lowest balance: {min_comp:.2f} on {min_comp_row['date']}")
print(f"Headroom above min balance ({p05['minimum_balance_to_keep']}): {min_comp - float(p05['minimum_balance_to_keep']):.2f}")

# Summary at full 90 days (2026-02-04, day 90)
min_90 = df['closing_balance'].min()
min_90_row = df.loc[df['closing_balance'].idxmin()]
print(f"\n--- AT FULL 90-DAY HORIZON (day {min_90_row['day']}) ---")
print(f"Lowest balance: {min_90:.2f} on {min_90_row['date']}")
print(f"Headroom above min balance ({p05['minimum_balance_to_keep']}): {min_90 - float(p05['minimum_balance_to_keep']):.2f}")
