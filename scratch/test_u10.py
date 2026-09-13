import sys
import os
sys.path.insert(0, os.path.abspath('.'))
import pandas as pd
import numpy as np

from code.forecasting import (
    EventNormalizer,
    ImageAmountResolver,
    ExchangeRateProvider,
    ConfirmedIncomeForecaster,
    ExpenseForecaster,
    DailyCashFlowSimulator,
)

profiles = pd.read_csv('dataset/financial_profiles.csv')
events = pd.read_csv('dataset/financial_events.csv')
sample_requests = pd.read_csv('dataset/sample_requests.csv')
messages = pd.read_csv('dataset/messages.csv')
rates = pd.read_csv('dataset/exchange_rates.csv')
images = pd.read_csv('dataset/images.csv')

fx = ExchangeRateProvider(rates)
norm = EventNormalizer(events, images, fx)
inc_f = ConfirmedIncomeForecaster(messages)
exp_f = ExpenseForecaster()

# Let's test user_10 without projecting gig income:
# If user_10 recurring salary is None:
u_events = norm.get_user_events('user_10', 'INR', '2024-12-06')
p10 = profiles[profiles['user_id'] == 'user_10'].iloc[0]
r10 = sample_requests[sample_requests['request_id'] == 'request_10'].iloc[0]

sim = DailyCashFlowSimulator(inc_f, exp_f, variable_spending_stat='median', forecast_horizon_days=90)
# Mocking salary_schedule = None for user_10
res = sim.simulate(
    user_id='user_10',
    user_events=u_events,
    current_available_balance=float(p10['current_available_balance']),
    minimum_balance_to_keep=float(p10['minimum_balance_to_keep']),
    requested_amount=float(r10['requested_amount']),
    request_date='2024-12-06'
)

print("User 10 default with salary:", res.amount_safe_to_pay)
