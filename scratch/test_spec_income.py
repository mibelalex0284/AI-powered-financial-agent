"""
Test ONLY the specification-supported general corrections:
1. Message payroll date updates (e.g. message_05 for user_07 moving payday from 15th to 23rd).
2. Base salary stream hygiene (exclude bonuses/commissions/second income from base salary cadence detection).
3. Test across all 25 samples.
"""

import os
import sys
sys.path.insert(0, os.path.abspath('.'))

import pandas as pd
import numpy as np
import re
from datetime import datetime
from typing import Dict, List, Optional, Tuple, Set

profiles = pd.read_csv('dataset/financial_profiles.csv')
events = pd.read_csv('dataset/financial_events.csv')
rates = pd.read_csv('dataset/exchange_rates.csv')
sample_requests = pd.read_csv('dataset/sample_requests.csv')
options = pd.read_csv('dataset/request_payment_options.csv')
messages = pd.read_csv('dataset/messages.csv')
images = pd.read_csv('dataset/images.csv')

from code.forecasting import (
    EventNormalizer,
    ExchangeRateProvider,
    ConfirmedIncomeForecaster,
    ExpenseForecaster,
    DailyCashFlowSimulator,
    RecurringIncomeSchedule,
)
from code.optimization.spending_changes import SpendingChangeOptimizer
from code.optimization.payment_optimizer import Payment, CandidatePlan, PlanSafetyEvaluator
from code.optimization.decision_engine import DecisionEngine

fx = ExchangeRateProvider(rates)
norm = EventNormalizer(events, images, fx)

class SpecIncomeForecaster(ConfirmedIncomeForecaster):
    def __init__(self, messages_df: Optional[pd.DataFrame] = None):
        super().__init__(messages_df)
        self.payroll_day_overrides: Dict[str, int] = {}
        self._parse_payroll_day_messages(messages_df)

    def _parse_payroll_day_messages(self, messages_df: Optional[pd.DataFrame]):
        """Parse employer messages that revise the salary date (§6.3 amendment rule)."""
        if messages_df is None or messages_df.empty:
            return
        for _, row in messages_df.iterrows():
            uid = str(row['user_id']).strip()
            txt = str(row.get('message_text', ''))
            # Example: "Your confirmed salary is now expected on 2024-09-23. This replaces the payroll date..."
            m = re.search(r'confirmed salary is now expected on (\d{4}-\d{2}-\d{2})', txt)
            if m:
                target_d = datetime.strptime(m.group(1), "%Y-%m-%d")
                self.payroll_day_overrides[uid] = target_d.day

    def get_recurring_salary_schedule(self, user_id: str, user_events: pd.DataFrame, request_date: str) -> Optional[RecurringIncomeSchedule]:
        if not self.has_future_confirmed_income(user_id, user_events):
            return None

        override_amt = self.salary_overrides.get(user_id)
        req_d = str(request_date)[:10]

        # Settle history before request_date
        hist_sal = user_events[
            (user_events['category'] == 'salary')
            & (user_events['status'] == 'settled')
            & (user_events['direction'] == 'credit')
            & (user_events['settlement_date'] < req_d)
        ].copy()

        # Clean salary stream: exclude commissions, bonuses, and secondary income per §6.3
        if not hist_sal.empty:
            non_base_mask = hist_sal['description'].str.lower().str.contains('bonus|commission|second household|arrears')
            base_sal = hist_sal[~non_base_mask]
            if not base_sal.empty:
                hist_sal = base_sal

        if hist_sal.empty:
            sched_sal = user_events[
                (user_events['category'] == 'salary')
                & (user_events['status'] == 'scheduled')
                & (user_events['direction'] == 'credit')
                & (user_events['settlement_date'] >= req_d)
            ]
            if not sched_sal.empty:
                s_row = sched_sal.iloc[0]
                s_date = datetime.strptime(str(s_row['settlement_date'])[:10], "%Y-%m-%d")
                amt = override_amt if override_amt is not None else float(s_row['home_amount'])
                day = self.payroll_day_overrides.get(user_id, s_date.day)
                return RecurringIncomeSchedule(cadence='monthly', amount=amt, day_of_month=day)
            return None

        hist_sal = hist_sal.sort_values('settlement_date')
        settle_dates = pd.to_datetime(hist_sal['settlement_date'])
        diffs = settle_dates.diff().dt.days.dropna()
        med_diff = float(diffs.median()) if not diffs.empty else 30.0

        if override_amt is not None:
            amt = override_amt
        else:
            amts = hist_sal['home_amount'].dropna()
            amt = float(amts.median()) if not amts.empty else 0.0

        # Day of month override from message takes precedence (§6.3)
        if user_id in self.payroll_day_overrides:
            return RecurringIncomeSchedule(cadence='monthly', amount=amt, day_of_month=self.payroll_day_overrides[user_id])

        # Weekly cadence (5 to 9 days)
        if 5.0 <= med_diff <= 9.0:
            last_date = settle_dates.iloc[-1]
            return RecurringIncomeSchedule(cadence='weekly', amount=amt, day_of_week=last_date.dayofweek, anchor_date=str(last_date)[:10])
        # Biweekly cadence (12 to 16 days)
        elif 12.0 <= med_diff <= 16.0:
            last_date = settle_dates.iloc[-1]
            return RecurringIncomeSchedule(cadence='biweekly', amount=amt, anchor_date=str(last_date)[:10])
        # Monthly cadence (25 to 35 days)
        elif 25.0 <= med_diff <= 35.0:
            typ_day = int(settle_dates.dt.day.mode().iloc[0])
            return RecurringIncomeSchedule(cadence='monthly', amount=amt, day_of_month=typ_day)
        elif len(hist_sal) >= 3 and settle_dates.dt.day.nunique() <= 2:
            typ_day = int(settle_dates.dt.day.mode().iloc[0])
            return RecurringIncomeSchedule(cadence='monthly', amount=amt, day_of_month=typ_day)

        typ_day = int(settle_dates.dt.day.mode().iloc[0])
        return RecurringIncomeSchedule(cadence='monthly', amount=amt, day_of_month=typ_day)


# Initialize and test
inc_f_spec = SpecIncomeForecaster(messages)
exp_f_base = ExpenseForecaster()

engine_spec = DecisionEngine(
    event_normalizer=norm,
    income_forecaster=inc_f_spec,
    expense_forecaster=exp_f_base,
    variable_spending_stat='median',
)

profiles_map = {str(p['user_id']).strip(): p for _, p in profiles.iterrows()}

status_matches = 0
method_matches = 0
plan_matches = 0
earliest_matches = 0
spending_matches = 0
exact_safe_matches = 0

abs_errors = []
norm_errors = []

for _, row in sample_requests.iterrows():
    r_id = str(row['request_id']).strip()
    u_id = str(row['user_id']).strip()
    p = profiles_map[u_id]

    res = engine_spec.evaluate_request(row, p, options)

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
    if gt_status == res.affordability_status:
        status_matches += 1

    gt_method = str(row['recommended_payment_method']).strip()
    if gt_method == res.recommended_payment_method:
        method_matches += 1

    gt_plan = str(row['payment_plan']).strip()
    if gt_plan == res.payment_plan:
        plan_matches += 1

    gt_earliest = str(row['earliest_date_for_full_payment']).strip() if pd.notna(row['earliest_date_for_full_payment']) else ""
    if gt_earliest == res.earliest_date_for_full_payment:
        earliest_matches += 1

    gt_changes = str(row['spending_changes_needed']).strip()
    if gt_changes == res.spending_changes_needed:
        spending_matches += 1

    if r_id in ['request_07', 'request_13']:
        print(f"{r_id}: Pred Earliest={res.earliest_date_for_full_payment}, GT Earliest={gt_earliest} | Pred Safe={pred_safe}, GT Safe={gt_safe} | Method={res.recommended_payment_method}, GT Method={gt_method}")

n = len(sample_requests)
print("=" * 70)
print(f"Status Accuracy:          {status_matches} / {n} ({status_matches/n*100:.1f}%)")
print(f"Payment Method Accuracy:  {method_matches} / {n} ({method_matches/n*100:.1f}%)")
print(f"Payment Plan Accuracy:    {plan_matches} / {n} ({plan_matches/n*100:.1f}%)")
print(f"Spending Changes Accuracy:{spending_matches} / {n} ({spending_matches/n*100:.1f}%)")
print(f"Earliest Date Accuracy:   {earliest_matches} / {n} ({earliest_matches/n*100:.1f}%)")
print(f"Normalized Error (Avg):   {np.mean(norm_errors)*100:.2f}%")
print(f"MAE:                      {np.mean(abs_errors):,.2f}")
print("=" * 70)
