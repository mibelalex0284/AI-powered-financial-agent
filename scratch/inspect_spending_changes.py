import pandas as pd

events = pd.read_csv('dataset/financial_events.csv')
profiles = pd.read_csv('dataset/financial_profiles.csv')

target_events = ['event_476', 'event_989', 'event_1815', 'event_1816']
sub = events[events['event_id'].isin(target_events)]
for idx, r in sub.iterrows():
    u_prof = profiles[profiles['user_id'] == r['user_id']].iloc[0]
    print(f"{r['event_id']}: user={r['user_id']}, cat={r['category']}, amt={r['amount']}, flex={r['flexibility']}, min_amt={r['minimum_allowed_amount']}, desc={r['description']}")
    print(f"   willing_stop: {u_prof['expense_categories_user_is_willing_to_stop']}")
    print(f"   willing_reduce: {u_prof['expense_categories_user_is_willing_to_reduce']}")
    print(f"   protected: {u_prof['expense_categories_to_protect']}\n")
