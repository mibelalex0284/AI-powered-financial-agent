import pandas as pd

df = pd.read_csv('scratch/r05_trace.csv')
key_dates = ['2025-11-06', '2025-11-15', '2025-12-01', '2025-12-02', '2025-12-15', 
             '2026-01-01', '2026-01-02', '2026-01-12', '2026-01-15', 
             '2026-01-20', '2026-01-25', '2026-02-01', '2026-02-02', '2026-02-03', '2026-02-04']

print("Key dates in r05 trace:")
for d in key_dates:
    sub = df[df['date'] == d]
    if not sub.empty:
        r = sub.iloc[0]
        print(f"{r['date']}: open={r['opening_balance']:9.2f}, rec={r['recurring_fixed_expenses']:7.2f}, var={r['variable_essential_spending']:6.2f}, close={r['closing_balance']:9.2f}")

# Look at rows where balance is around 13837 (which would leave 737)
print("\nDates where closing balance is closest to 13837:")
df['diff_13837'] = (df['closing_balance'] - 13837.0).abs()
closest = df.sort_values('diff_13837').head(5)
for _, r in closest.iterrows():
    print(f"Date {r['date']} (day {r['day']}): close={r['closing_balance']:.2f}, diff={r['diff_13837']:.2f}")
