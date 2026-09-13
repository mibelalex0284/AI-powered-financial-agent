import pandas as pd
samples = pd.read_csv('dataset/sample_requests.csv')
for rid in ['request_04', 'request_07', 'request_13', 'request_20', 'request_25']:
    print(samples[samples['request_id'] == rid].iloc[0])
    print('-'*50)
