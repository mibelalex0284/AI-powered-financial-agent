from datetime import datetime, timedelta

cur_bal = 218945.56
min_bal = 93000.0  
req_d = datetime(2024, 9, 4)

rent_day = 4; utilities_day = 8; debt_day = 13; music_day = 13; salary_day = 23
rent_amt = 34200.0; utilities_amt = 7049.68; debt_amt = 15650.0; music_amt = 1005.0; salary_amt = 149000.0

# Check the first instance: d.day == 4 on Sep 4 (day 0) means rent IS on day 0 (request_date itself)!
current_bal = cur_bal
min_b = current_bal
min_d = req_d.strftime('%Y-%m-%d')

for day in range(90):
    d = req_d + timedelta(days=day)
    d_str = d.strftime('%Y-%m-%d')
    inc = 0.0; exp = 0.0
    if d.day == rent_day: exp += rent_amt
    if d.day == utilities_day: exp += utilities_amt
    if d.day == debt_day: exp += debt_amt
    if d.day == music_day: exp += music_amt
    if d.day == salary_day: inc += salary_amt
    current_bal += inc - exp
    if current_bal < min_b:
        min_b = current_bal
        min_d = d_str
    if exp > 0 or inc > 0:
        label = d_str + ': inc=' + str(inc) + ', exp=' + str(exp) + ', bal=' + str(round(current_bal,2))
        print(label)

print('Min balance:', round(min_b, 2), 'on', min_d)
print('Safe today (fixed only):', round(min_b - min_bal, 2))
