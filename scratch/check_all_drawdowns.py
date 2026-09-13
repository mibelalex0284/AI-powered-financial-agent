import pandas as pd
import numpy as np

# Load profiles and sample requests
profiles = pd.read_csv('dataset/financial_profiles.csv')
sample_requests = pd.read_csv('dataset/sample_requests.csv')

df = sample_requests.merge(profiles, on='user_id')
print("Benchmark drawdown and safe amounts for all 25 samples:")
for idx, r in df.iterrows():
    curr = float(r['current_available_balance'])
    min_b = float(r['minimum_balance_to_keep'])
    safe = float(r['amount_safe_to_pay'])
    req = float(r['requested_amount'])
    
    # If safe > 0 and safe < req, then safe = projected_min - min_b
    # Therefore projected_min = min_b + safe
    # and total_drawdown = curr - projected_min = curr - min_b - safe
    drawdown = curr - min_b - safe
    print(f"{r['request_id']:10s} user={r['user_id']:7s} curr={curr:12.2f} min={min_b:12.2f} req={req:12.2f} safe={safe:12.2f} drawdown={drawdown:12.2f} comp_date={r['desired_completion_date']}")
