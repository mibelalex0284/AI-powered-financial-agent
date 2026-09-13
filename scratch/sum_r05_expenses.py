import pandas as pd

df = pd.read_csv('scratch/r05_trace.csv')
rec_days = df[df['recurring_fixed_expenses'] > 0]
print("All recurring expense days for r05:")
for _, r in rec_days.iterrows():
    print(f"Date {r['date']} (day {r['day']}): amount={r['recurring_fixed_expenses']:.2f}")

total_rec = df['recurring_fixed_expenses'].sum()
total_var = df['variable_essential_spending'].sum()
print(f"\nTotal Recurring over 90 days: {total_rec:.2f}")
print(f"Total Variable over 90 days: {total_var:.2f}")
print(f"Grand Total Outflows over 90 days: {total_rec + total_var:.2f}")
