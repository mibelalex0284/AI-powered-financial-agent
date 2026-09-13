import pandas as pd

events = pd.read_csv('dataset/financial_events.csv')
u23 = events[(events['user_id'] == 'user_23') & (events['status'] == 'settled')].copy()
u23['dt'] = pd.to_datetime(u23['settlement_date'])
print("Categories and counts for user_23:")
for cat, grp in u23.groupby('category'):
    print(f"{cat:15s}: count={len(grp)}, amounts={grp['amount'].tolist()[:5]}")

# Look at events that occurred between day 7 and day 14 in any past month for user_23!
print("\nHistorical events between day 7 and day 14 of each month for user_23:")
mid_month = u23[(u23['dt'].dt.day >= 7) & (u23['dt'].dt.day <= 14)].sort_values('dt')
for idx, r in mid_month.iterrows():
    print(f"{r['settlement_date']} ({r['direction']}): {r['category']:15s} amt={r['amount']:8.2f} desc={r['description']}")
