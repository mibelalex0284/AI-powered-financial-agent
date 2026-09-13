"""
Evaluate decision pipeline on 25 samples with enhanced general forecasting:
- Clean base salary detection (ignoring bonuses/commissions/second income)
- All recurring categories (dining, entertainment, shopping, plus fixed)
- Triweekly cadence (21-day) support
"""

import os
import sys
sys.path.insert(0, os.path.abspath('.'))

import pandas as pd
import numpy as np

from scratch.test_general_fixes import EnhancedIncomeForecaster, EnhancedExpenseForecaster, EnhancedDailyCashFlowSimulator
from code.optimization.spending_changes import SpendingChangeOptimizer
from code.optimization.payment_optimizer import Payment, CandidatePlan, PlanSafetyEvaluator
from code.optimization.decision_engine import DecisionEngine

# Load datasets
profiles = pd.read_csv('dataset/financial_profiles.csv')
events = pd.read_csv('dataset/financial_events.csv')
rates = pd.read_csv('dataset/exchange_rates.csv')
sample_requests = pd.read_csv('dataset/sample_requests.csv')
options = pd.read_csv('dataset/request_payment_options.csv')
messages = pd.read_csv('dataset/messages.csv')
images = pd.read_csv('dataset/images.csv')

from code.forecasting import EventNormalizer, ExchangeRateProvider
fx = ExchangeRateProvider(rates)
norm = EventNormalizer(events, images, fx)

inc_f = EnhancedIncomeForecaster(messages)
exp_f = EnhancedExpenseForecaster()

# Create decision engine
engine = DecisionEngine(
    event_normalizer=norm,
    income_forecaster=inc_f,
    expense_forecaster=exp_f,
    variable_spending_stat='median',
)
# Plug in enhanced simulator
engine.simulator = EnhancedDailyCashFlowSimulator(inc_f, exp_f, 'median', 90)

profiles_map = {str(p['user_id']).strip(): p for _, p in profiles.iterrows()}

status_matches = 0
method_matches = 0
plan_matches = 0
earliest_matches = 0
spending_matches = 0
exact_safe_matches = 0

abs_errors = []
norm_errors = []

print("=" * 110)
print(f"{'Req ID':<10} | {'Pred Safe':<12} | {'GT Safe':<12} | {'Status (P/GT)':<26} | {'Method (P/GT)':<24} | {'Changes (P/GT)':<30}")
print("-" * 110)

for _, row in sample_requests.iterrows():
    r_id = str(row['request_id']).strip()
    u_id = str(row['user_id']).strip()
    p = profiles_map[u_id]

    res = engine.evaluate_request(row, p, options)

    gt_safe = float(row['amount_safe_to_pay'])
    pred_safe = float(res.amount_safe_to_pay)
    abs_err = abs(pred_safe - gt_safe)
    req_amt = float(row['requested_amount'])
    norm_err = abs_err / req_amt if req_amt > 0 else 0.0

    abs_errors.append(abs_err)
    norm_errors.append(norm_err)

    if abs_err < 0.01:
        exact_safe_matches += 1

    gt_status = str(row['affordability_status']).strip()
    pred_status = res.affordability_status
    if gt_status == pred_status:
        status_matches += 1

    gt_method = str(row['recommended_payment_method']).strip()
    pred_method = res.recommended_payment_method
    if gt_method == pred_method:
        method_matches += 1

    gt_plan = str(row['payment_plan']).strip()
    pred_plan = res.payment_plan
    if gt_plan == pred_plan:
        plan_matches += 1

    gt_earliest = str(row['earliest_date_for_full_payment']).strip() if pd.notna(row['earliest_date_for_full_payment']) else ""
    pred_earliest = res.earliest_date_for_full_payment
    if gt_earliest == pred_earliest:
        earliest_matches += 1

    gt_changes = str(row['spending_changes_needed']).strip()
    pred_changes = res.spending_changes_needed
    if gt_changes == pred_changes:
        spending_matches += 1

    status_str = f"{pred_status[:12]} / {gt_status[:12]}"
    method_str = f"{pred_method[:11]} / {gt_method[:11]}"
    changes_str = f"{pred_changes[:14]} / {gt_changes[:14]}"

    print(f"{r_id:<10} | {pred_safe:<12.2f} | {gt_safe:<12.2f} | {status_str:<26} | {method_str:<24} | {changes_str:<30}")

n = len(sample_requests)
print("=" * 110)
print(f"Status Accuracy:          {status_matches} / {n} ({status_matches/n*100:.1f}%)")
print(f"Payment Method Accuracy:  {method_matches} / {n} ({method_matches/n*100:.1f}%)")
print(f"Payment Plan Accuracy:    {plan_matches} / {n} ({plan_matches/n*100:.1f}%)")
print(f"Spending Changes Accuracy:{spending_matches} / {n} ({spending_matches/n*100:.1f}%)")
print(f"Earliest Date Accuracy:   {earliest_matches} / {n} ({earliest_matches/n*100:.1f}%)")
print(f"Normalized Error (Avg):   {np.mean(norm_errors)*100:.2f}%")
print(f"MAE:                      {np.mean(abs_errors):,.2f}")
print("=" * 110)
