import pandas as pd

events = pd.read_csv('dataset/financial_events.csv')
u11 = events[(events['user_id'] == 'user_11') & (events['direction'] == 'debit') & (events['status'] == 'settled')]
print('u11 categories and counts:')
for cat, g in u11.groupby('category'):
    print(f"{cat}: count={len(g)}, median={g['amount'].median():,.2f}, last_settle={g['settlement_date'].max()}")
