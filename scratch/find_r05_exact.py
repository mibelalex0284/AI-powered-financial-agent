import sys
import os
sys.path.insert(0, os.path.abspath('.'))
import pandas as pd
import numpy as np

target_outflow = 32638.10

# We want to find what subset of recurring expenses and what variable spending
# sums to 32638.10, or what combination of days/quantiles/commitments equals 32638.10.

events = pd.read_csv('dataset/financial_events.csv')
ev05 = events[(events['user_id'] == 'user_05') & (events['status'] == 'settled')]

# Let's inspect all categories and amounts
print("Historical expenses for user_05 by category:")
cats = {}
for cat, grp in ev05.groupby('category'):
    if cat != 'salary':
        cats[cat] = {
            'amounts': grp['amount'].tolist(),
            'mean': grp['amount'].mean(),
            'median': grp['amount'].median(),
            'count': len(grp)
        }
        print(f"{cat:15s}: count={len(grp)}, median={grp['amount'].median():.2f}, mean={grp['amount'].mean():.2f}, sum={grp['amount'].sum():.2f}")

# Let's test combinations of fixed monthly expenses and variable spending
# Monthly categories:
# rent: 4972.0
# debt_repayment: 968.0
# family_support: 840.4
# cloud_storage: 113.3
# utilities: median 706.37 or mean 686.71
# healthcare: median 721.44 or mean 699.01
# shopping: median 404.24 or mean 397.85

# Notice: In profiles.csv:
# expense_categories_to_protect: rent|healthcare|family_support|groceries
# expense_categories_user_is_willing_to_reduce: shopping
# expense_categories_user_is_willing_to_stop: cloud_storage

# Groceries & Transport weekly stats:
# Weekly groceries:
# weekly transport:
# Let's check what the weekly numbers are:
from code.forecasting import ExpenseForecaster, EventNormalizer, ImageAmountResolver, ExchangeRateProvider

fx = ExchangeRateProvider(pd.read_csv('dataset/exchange_rates.csv'))
norm = EventNormalizer(pd.read_csv('dataset/financial_events.csv'), pd.read_csv('dataset/images.csv'), fx)
u_events = norm.get_user_events('user_05', 'ZAR', '2025-11-06')

exp_f = ExpenseForecaster()
stats = exp_f.get_variable_essential_stats(u_events, '2025-11-06')
print("Essential spend weekly stats:", stats)

# If horizon is 90 days: 90 / 7 = 12.857 weeks
# If horizon is 67 days (to 2026-01-12): 67 / 7 = 9.57 weeks
# What if horizon is 2 months (60 days)?
# What if horizon is 3 months (90 days)?

for weeks in [90/7, 67/7, 12, 13, 9, 10]:
    for stat_name, w_val in stats.items():
        var_tot = weeks * w_val
        rem = target_outflow - var_tot
        print(f"weeks={weeks:.2f}, stat={stat_name} (val={w_val:.2f}): var_tot={var_tot:.2f} -> remaining fixed needed={rem:.2f}")

