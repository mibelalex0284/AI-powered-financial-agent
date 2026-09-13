import pandas as pd

samples = pd.read_csv('dataset/sample_requests.csv')
for idx, r in samples.iterrows():
    print(f"{r['request_id']}: status={r['affordability_status']}, method={r['recommended_payment_method']}, safe={r['amount_safe_to_pay']}, earliest={r['earliest_date_for_full_payment']}, plan={r['payment_plan']}, changes={r['spending_changes_needed']}")
    print(f"   explanation: {r['decision_explanation']}\n")
