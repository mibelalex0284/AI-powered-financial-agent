import pandas as pd

samples = pd.read_csv('dataset/sample_requests.csv')
options = pd.read_csv('dataset/request_payment_options.csv')

inst_samples = samples[samples['recommended_payment_method'] == 'installments']
print("Installment samples in sample_requests.csv:")
for idx, r in inst_samples.iterrows():
    rid = r['request_id']
    print(f"Request: {rid}")
    print(f"  Sample plan: {r['payment_plan']}")
    print(f"  Available options:")
    opts = options[options['request_id'] == rid]
    for o_idx, o in opts.iterrows():
        print(f"    {o['payment_option_id']}: method={o['payment_method']}, amt={o['payment_amount']}, n={o['number_of_payments']}, start={o['first_payment_date']}, freq={o['payment_frequency_days']}, fee={o['financing_fee']}, total={o['total_payable_amount']}")
    print()
