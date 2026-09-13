import pandas as pd

events = pd.read_csv('dataset/financial_events.csv')
sample_requests = pd.read_csv('dataset/sample_requests.csv')

print("Salary descriptions for all sample users:")
for idx, req in sample_requests.iterrows():
    uid = req['user_id']
    u_sal = events[(events['user_id'] == uid) & (events['category'] == 'salary')]
    descs = u_sal['description'].unique().tolist()
    print(f"{req['request_id']} ({uid}): {descs}")
