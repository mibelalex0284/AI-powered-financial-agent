import pandas as pd
import numpy as np
from datetime import datetime, timedelta

events = pd.read_csv('dataset/financial_events.csv')
profiles = pd.read_csv('dataset/financial_profiles.csv')
samples = pd.read_csv('dataset/sample_requests.csv')

def analyze_user(req_id, uid, req_d, gt_safe):
    prof = profiles[profiles['user_id'] == uid].iloc[0]
    curr_b = float(prof['current_available_balance'])
    min_b = float(prof['minimum_balance_to_keep'])
    
    u_events = events[(events['user_id'] == uid) & (events['status'] == 'settled') & (events['settlement_date'] < req_d)]
    
    # Headroom
    headroom = curr_b - min_b
    implied_burn = headroom - gt_safe
    print(f"\n==================== {req_id} ({uid}) ====================")
    print(f"Current: {curr_b}, Min: {min_b}, Headroom: {headroom}")
    print(f"GT Safe: {gt_safe}")
    print(f"Implied burn to min_bal date: {implied_burn:.2f}")

analyze_user('request_06', 'user_06', '2026-01-03', 603.30)
analyze_user('request_11', 'user_11', '2025-05-03', 12510645.0)
analyze_user('request_07', 'user_07', '2024-09-05', 87170.56)
analyze_user('request_13', 'user_13', '2024-03-07', 433.40)
analyze_user('request_21', 'user_21', '2026-04-03', 1543.35)
