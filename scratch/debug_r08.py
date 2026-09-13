import pandas as pd
from datetime import datetime, timedelta

df = pd.read_csv('dataset/financial_events.csv')
u08 = df[(df['user_id']=='user_08') & (df['direction']=='debit') & (df['status']=='settled')]
for cat in ['groceries', 'transport', 'dining']:
    events = u08[u08['category']==cat].sort_values('settlement_date')
    dates = pd.to_datetime(events['settlement_date'])
    diffs = dates.diff().dt.days.dropna()
    print(cat + ': median_diff=' + str(diffs.median()) + ', mean_amt=' + str(round(events['amount'].mean(),2)))
    print('  last_date: ' + events['settlement_date'].iloc[-1])

# Key question: why does our system say not_affordable for request_08?
# GT says: earliest_date = 2025-04-15 (April salary day)
# Our system can't find any safe payment date in 90 days

# Let me manually simulate: will April 15 payment (996.60) be safe?
# Starting balance: 1536.57 (no pending debits for user_08)
# Feb 7 to May 7 = 90 days

cur_bal = 1536.57
min_bal = 800.0
req_d = datetime(2025, 2, 7)

# Fixed monthly expenses
# Using medians from history
rent = 467.50  # consistent
utils = 80.55  # recent
edu = 89.00
debt = 177.00
music = 14.00
delivery = 24.00
salary = 1422.85

# Daily variable for groceries + transport
# From events: groceries weekly ~55/week = 7.9/day, transport weekly ~37/week = 5.3/day  
# dining biweekly ~50/2weeks = 3.6/day
# But these are discrete recurring events in our system
# Let's use a simple daily var = 0 and model them as discrete

# Groceries: weekly anchor 2025-02-04 (last before request)
groc_anchor = datetime(2025, 2, 4)
groc_amt = 55.0  # approximate
trans_anchor = datetime(2025, 2, 5)  # last transport was 2025-02-05
trans_amt = 38.0
dining_anchor = datetime(2025, 1, 30)  # last dining was 2025-01-30
dining_amt = 48.0

current_bal = cur_bal
min_b = current_bal
min_d = req_d.strftime('%Y-%m-%d')
payment_date = datetime(2025, 4, 15)
payment_made = False

print("\nSIMULATING with April 15 payment:")
for day in range(90):
    d = req_d + timedelta(days=day)
    d_str = d.strftime('%Y-%m-%d')
    inc = 0.0; exp = 0.0
    # Fixed monthly
    if d.day == 1: exp += rent
    if d.day == 5: exp += utils
    if d.day == 7: exp += edu
    if d.day == 10: exp += debt + music
    if d.day == 12: exp += delivery
    if d.day == 15: inc += salary
    # Payment on April 15
    if d == payment_date and not payment_made:
        exp += 996.60
        payment_made = True
    current_bal += inc - exp
    if current_bal < min_b:
        min_b = current_bal
        min_d = d_str
    if exp > 0 or inc > 0:
        print(d_str + ': inc=' + str(inc) + ', exp=' + str(round(exp,2)) + ', bal=' + str(round(current_bal,2)))

print('Min balance: ' + str(round(min_b,2)) + ' on ' + min_d)
print('Is safe: ' + str(min_b >= min_bal))
