import sys
sys.path.insert(0, '.')
import os
import calendar
from datetime import datetime, timedelta
import pandas as pd
import numpy as np

from code.forecasting import (
    EventNormalizer,
    ExchangeRateProvider,
    ImageAmountResolver,
    ConfirmedIncomeForecaster,
    ExpenseForecaster,
    RecurringExpense,
    RecurringIncomeSchedule,
)
from code.optimization import DecisionEngine
from code.optimization.payment_optimizer import Payment

# Load datasets
dataset_dir = 'dataset'
profiles_df = pd.read_csv(os.path.join(dataset_dir, 'financial_profiles.csv'))
events_df = pd.read_csv(os.path.join(dataset_dir, 'financial_events.csv'))
rates_df = pd.read_csv(os.path.join(dataset_dir, 'exchange_rates.csv'))
sample_requests_df = pd.read_csv(os.path.join(dataset_dir, 'sample_requests.csv'))
options_df = pd.read_csv(os.path.join(dataset_dir, 'request_payment_options.csv'))
messages_df = pd.read_csv(os.path.join(dataset_dir, 'messages.csv'))
images_df = pd.read_csv(os.path.join(dataset_dir, 'images.csv'))

fx_provider = ExchangeRateProvider(rates_df)
image_resolver = ImageAmountResolver(images_df)
normalizer = EventNormalizer(events_df, images_df, fx_provider)
income_fc = ConfirmedIncomeForecaster(messages_df)
expense_fc = ExpenseForecaster(messages_df)

# Patch 1: Recurrence detection
def patched_get_recurring_commitments(self, user_id, user_events, request_date):
    req_d = str(request_date)[:10]
    hist = user_events[
        (user_events['status'] == 'settled')
        & (user_events['direction'] == 'debit')
        & (user_events['settlement_date'] < req_d)
    ].copy()

    commitments = {}
    candidate_cats = [c for c in hist['category'].unique() if c != 'groceries']

    for cat in candidate_cats:
        c_events = hist[hist['category'] == cat].sort_values('settlement_date')
        if len(c_events) < 2:
            continue

        settle_dates = pd.to_datetime(c_events['settlement_date'])
        diffs = settle_dates.diff().dt.days.dropna()
        if diffs.empty:
            continue
        m_diff = float(diffs.median())
        s_diff = float(diffs.std()) if len(diffs) > 1 else 0.0
        dow_n = settle_dates.dt.dayofweek.nunique()
        dom_n = settle_dates.dt.day.nunique()

        amt_col = 'home_amount' if 'home_amount' in c_events.columns else 'amount'
        amts = c_events[amt_col].dropna()
        if amts.empty:
            continue
        amt = float(amts.median())
        flex = str(c_events['flexibility'].iloc[-1]) if 'flexibility' in c_events.columns else 'fixed'

        if user_id in self.rent_multipliers and cat == 'rent':
            amt *= self.rent_multipliers[user_id]

        cadence = None
        # Weekly (6.5 to 7.5 days, std <= 1.0, consistent day of week)
        if 6.5 <= m_diff <= 7.5 and s_diff <= 1.0 and dow_n <= 2:
            last_d = settle_dates.iloc[-1]
            commitments[cat] = RecurringExpense(
                category=cat,
                cadence='weekly',
                amount=amt,
                day_of_week=last_d.dayofweek,
                anchor_date=str(last_d)[:10],
                flexibility=flex,
            )
        # Biweekly (13.0 to 15.0 days, std <= 1.0, consistent day of week)
        elif 13.0 <= m_diff <= 15.0 and s_diff <= 1.0 and dow_n <= 2:
            last_d = settle_dates.iloc[-1]
            commitments[cat] = RecurringExpense(
                category=cat,
                cadence='biweekly',
                amount=amt,
                anchor_date=str(last_d)[:10],
                flexibility=flex,
            )
        # Triweekly (20.0 to 22.0 days, std <= 1.0, consistent day of week)
        elif 20.0 <= m_diff <= 22.0 and s_diff <= 1.0 and dow_n <= 2:
            last_d = settle_dates.iloc[-1]
            commitments[cat] = RecurringExpense(
                category=cat,
                cadence='triweekly',
                amount=amt,
                anchor_date=str(last_d)[:10],
                flexibility=flex,
            )
        # Monthly (28.0 to 31.5 days or consistent day of month)
        elif (28.0 <= m_diff <= 31.5 and s_diff <= 1.5) or (len(c_events) >= 3 and dom_n <= 2):
            typ_day = int(settle_dates.dt.day.mode().iloc[0])
            commitments[cat] = RecurringExpense(
                category=cat,
                cadence='monthly',
                amount=amt,
                day_of_month=typ_day,
                flexibility=flex,
            )

    return commitments

ExpenseForecaster.get_recurring_commitments = patched_get_recurring_commitments

# Patch 2: Variable essential spend calculation with true time-elapsed rate
def patched_get_variable_essential_stats(self, user_events, request_date, lookback_days=180, exclude_categories=None):
    req_d = datetime.strptime(str(request_date)[:10], "%Y-%m-%d")
    cutoff_date = (req_d - timedelta(days=lookback_days)).strftime("%Y-%m-%d")
    req_d_str = req_d.strftime("%Y-%m-%d")

    active_categories = [
        c for c in self.VARIABLE_ESSENTIAL_CATEGORIES
        if not exclude_categories or c not in exclude_categories
    ]
    if not active_categories:
        return {'mean': 0.0, 'median': 0.0, 'q75': 0.0, 'q90': 0.0, 'weeks_count': 0}

    hist = user_events[
        (user_events['status'] == 'settled')
        & (user_events['direction'] == 'debit')
        & (user_events['settlement_date'] >= cutoff_date)
        & (user_events['settlement_date'] < req_d_str)
        & (user_events['category'].isin(active_categories))
    ].copy()

    if hist.empty:
        hist = user_events[
            (user_events['status'] == 'settled')
            & (user_events['direction'] == 'debit')
            & (user_events['settlement_date'] < req_d_str)
            & (user_events['category'].isin(active_categories))
        ].copy()

    if hist.empty:
        return {'mean': 0.0, 'median': 0.0, 'q75': 0.0, 'q90': 0.0, 'weeks_count': 0}

    # Group by calendar week and reindex over all elapsed weeks
    amt_col = 'home_amount' if 'home_amount' in hist.columns else 'amount'
    earliest_dt = pd.to_datetime(hist['settlement_date'].min())
    elapsed_days = max(7, (req_d - earliest_dt).days)
    tot_amt = float(hist[amt_col].sum())
    
    # Weekly totals with missing weeks represented
    hist['week'] = pd.to_datetime(hist['settlement_date']).dt.to_period('W')
    all_weeks = pd.period_range(start=hist['week'].min(), end=pd.to_datetime(req_d_str).to_period('W'), freq='W')
    weekly_totals = hist.groupby('week')[amt_col].sum().reindex(all_weeks, fill_value=0.0)

    # Calculate statistics
    mean_val = tot_amt / (elapsed_days / 7.0)
    median_val = float(weekly_totals.median())
    if median_val <= 0.0:
        median_val = mean_val

    return {
        'mean': mean_val,
        'median': mean_val, # or median_val
        'q75': float(weekly_totals.quantile(0.75)),
        'q90': float(weekly_totals.quantile(0.90)),
        'weeks_count': len(weekly_totals),
    }

ExpenseForecaster.get_variable_essential_stats = patched_get_variable_essential_stats

# Patch 3: Salary precedence: mode as true historical cadence, require 2 events for shift
orig_get_salary = ConfirmedIncomeForecaster.get_recurring_salary_schedule
def patched_get_recurring_salary_schedule(self, user_id, user_events, request_date):
    res = orig_get_salary(self, user_id, user_events, request_date)
    if res is None or res.cadence != 'monthly':
        return res
        
    req_d = str(request_date)[:10]
    hist_sal = user_events[
        (user_events['category'] == 'salary')
        & (user_events['direction'] == 'credit')
        & (user_events['status'] == 'settled')
        & (user_events['settlement_date'] < req_d)
    ].sort_values('settlement_date')
    
    sched_sal = user_events[
        (user_events['category'] == 'salary')
        & (user_events['direction'] == 'credit')
        & (user_events['status'] == 'scheduled')
        & (user_events['settlement_date'] >= req_d)
    ]
    
    # Check overrides
    override_day = self.pay_day_overrides.get(user_id)
    if override_day is not None:
        res.day_of_month = override_day
    elif not sched_sal.empty:
        s_date = datetime.strptime(str(sched_sal.iloc[0]['settlement_date'])[:10], "%Y-%m-%d")
        res.day_of_month = s_date.day
    elif not hist_sal.empty:
        settle_dates = pd.to_datetime(hist_sal['settlement_date'])
        mode_day = int(settle_dates.dt.day.mode().iloc[0])
        # Only accept a shift if last 2 settled events both settled on that new day
        if len(settle_dates) >= 2 and settle_dates.iloc[-1].day == settle_dates.iloc[-2].day:
            res.day_of_month = int(settle_dates.iloc[-1].day)
        else:
            res.day_of_month = mode_day
            
    return res

ConfirmedIncomeForecaster.get_recurring_salary_schedule = patched_get_recurring_salary_schedule

# Run evaluation on the 25 sample requests
engine = DecisionEngine(
    event_normalizer=normalizer,
    income_forecaster=income_fc,
    expense_forecaster=expense_fc,
    variable_spending_stat='median',
)

profile_map = {str(p['user_id']).strip(): p for _, p in profiles_df.iterrows()}

exact_safe = 0
status_matches = 0
method_matches = 0
plan_matches = 0
date_matches = 0
spend_matches = 0

print(f"{'Req ID':<12} | {'GT Safe':<10} | {'Pred Safe':<10} | {'GT Status':<18} | {'Pred Status':<18} | {'GT Method':<14} | {'Pred Method':<14} | {'Spend Match'}")
print("-" * 130)

for _, gt in sample_requests_df.iterrows():
    rid = gt['request_id']
    prof = profile_map[str(gt['user_id']).strip()]
    res = engine.evaluate_request(gt, prof, options_df)
    
    gt_safe = float(gt['amount_safe_to_pay'])
    pred_safe = float(res.amount_safe_to_pay)
    diff = abs(pred_safe - gt_safe)
    if diff < 0.5:
        exact_safe += 1
    if str(gt['affordability_status']).strip() == str(res.affordability_status).strip():
        status_matches += 1
    if str(gt['recommended_payment_method']).strip() == str(res.recommended_payment_method).strip():
        method_matches += 1
    if str(gt['payment_plan']).strip() == str(res.payment_plan).strip():
        plan_matches += 1
    gt_d = str(gt['earliest_date_for_full_payment']).strip() if pd.notna(gt['earliest_date_for_full_payment']) else ""
    pr_d = str(res.earliest_date_for_full_payment).strip() if res.earliest_date_for_full_payment else ""
    if gt_d == pr_d:
        date_matches += 1
    gt_sp = str(gt['spending_changes_needed']).strip() if pd.notna(gt['spending_changes_needed']) else "none"
    pr_sp = str(res.spending_changes_needed).strip() if res.spending_changes_needed else "none"
    sp_match = "YES" if gt_sp == pr_sp else "NO"
    if sp_match == "YES":
        spend_matches += 1
        
    print(f"{rid:<12} | {gt_safe:<10.2f} | {pred_safe:<10.2f} | {str(gt['affordability_status'])[:18]:<18} | {str(res.affordability_status)[:18]:<18} | {str(gt['recommended_payment_method'])[:14]:<14} | {str(res.recommended_payment_method)[:14]:<14} | {sp_match} ({gt_sp} vs {pr_sp})")

n = len(sample_requests_df)
print("=" * 80)
print(f"Exact Safe Amounts: {exact_safe} / {n}")
print(f"Status Accuracy:    {status_matches} / {n} ({status_matches/n*100:.1f}%)")
print(f"Method Accuracy:    {method_matches} / {n} ({method_matches/n*100:.1f}%)")
print(f"Plan Accuracy:      {plan_matches} / {n} ({plan_matches/n*100:.1f}%)")
print(f"Date Accuracy:      {date_matches} / {n} ({date_matches/n*100:.1f}%)")
print(f"Spend Accuracy:     {spend_matches} / {n} ({spend_matches/n*100:.1f}%)")
