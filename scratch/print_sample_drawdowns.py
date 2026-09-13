import pandas as pd
samples = pd.read_csv('dataset/sample_requests.csv')
profiles = pd.read_csv('dataset/financial_profiles.csv')
merged = pd.merge(samples, profiles, on='user_id')

for _, r in merged.iterrows():
    curr = float(r['current_available_balance'])
    min_b = float(r['minimum_balance_to_keep'])
    req_amt = float(r['requested_amount'])
    safe = float(r['amount_safe_to_pay'])
    headroom = curr - min_b
    drawdown = headroom - safe
    print(f"{r['request_id']} | {r['user_id']} | Curr={curr:12.2f} | Min={min_b:10.2f} | Headroom={headroom:12.2f} | Req={req_amt:12.2f} | Safe={safe:12.2f} | Drawdown={drawdown:12.2f} | Status={r['affordability_status']}")
