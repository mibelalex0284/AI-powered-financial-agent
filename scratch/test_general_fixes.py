"""
Test the general, spec-derived forecasting corrections on the 25 samples:
1. Clean salary stream selection (filter out bonuses/commissions/secondary income from primary base salary).
2. Comprehensive recurring expense categories (include dining, entertainment, shopping).
3. Triweekly (21-day) cadence support for periodic dining/commitments.
"""

import os
import sys
sys.path.insert(0, os.path.abspath('.'))

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple, Set

from code.forecasting import (
    EventNormalizer,
    ExchangeRateProvider,
    ImageAmountResolver,
    ConfirmedIncomeForecaster,
    ExpenseForecaster,
    RecurringExpense,
    RecurringIncomeSchedule,
    DailyCashFlowSimulator,
)

# Load data
profiles = pd.read_csv('dataset/financial_profiles.csv')
events = pd.read_csv('dataset/financial_events.csv')
rates = pd.read_csv('dataset/exchange_rates.csv')
sample_requests = pd.read_csv('dataset/sample_requests.csv')
options = pd.read_csv('dataset/request_payment_options.csv')
messages = pd.read_csv('dataset/messages.csv')
images = pd.read_csv('dataset/images.csv')

fx = ExchangeRateProvider(rates)
norm = EventNormalizer(events, images, fx)

class EnhancedIncomeForecaster(ConfirmedIncomeForecaster):
    def get_recurring_salary_schedule(self, user_id: str, user_events: pd.DataFrame, request_date: str) -> Optional[RecurringIncomeSchedule]:
        if not self.has_future_confirmed_income(user_id, user_events):
            return None

        override_amt = self.salary_overrides.get(user_id)
        req_d = str(request_date)[:10]

        hist_sal = user_events[
            (user_events['category'] == 'salary')
            & (user_events['status'] == 'settled')
            & (user_events['direction'] == 'credit')
            & (user_events['settlement_date'] < req_d)
        ].copy()

        # Exclude bonuses, commissions, arrears, and second incomes
        # Per §6.3: Do not count bonuses, commissions until they settle (not as recurring salary)
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
                return RecurringIncomeSchedule(cadence='monthly', amount=amt, day_of_month=s_date.day)
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

        # Weekly
        if 5.0 <= med_diff <= 9.0:
            last_date = settle_dates.iloc[-1]
            return RecurringIncomeSchedule(cadence='weekly', amount=amt, day_of_week=last_date.dayofweek, anchor_date=str(last_date)[:10])
        # Biweekly
        elif 12.0 <= med_diff <= 16.0:
            last_date = settle_dates.iloc[-1]
            return RecurringIncomeSchedule(cadence='biweekly', amount=amt, anchor_date=str(last_date)[:10])
        # Monthly
        elif 25.0 <= med_diff <= 35.0:
            typ_day = int(settle_dates.dt.day.mode().iloc[0])
            return RecurringIncomeSchedule(cadence='monthly', amount=amt, day_of_month=typ_day)
        elif len(hist_sal) >= 3 and settle_dates.dt.day.nunique() <= 2:
            typ_day = int(settle_dates.dt.day.mode().iloc[0])
            return RecurringIncomeSchedule(cadence='monthly', amount=amt, day_of_month=typ_day)

        # Fallback monthly
        typ_day = int(settle_dates.dt.day.mode().iloc[0])
        return RecurringIncomeSchedule(cadence='monthly', amount=amt, day_of_month=typ_day)


class EnhancedExpenseForecaster(ExpenseForecaster):
    ALL_RECURRING_CATEGORIES = [
        'rent', 'utilities', 'debt_repayment', 'education', 'cloud_storage',
        'streaming', 'music_subscription', 'delivery_membership', 'gym',
        'insurance', 'family_support', 'housing', 'healthcare',
        'dining', 'entertainment', 'shopping'
    ]

    def get_recurring_commitments(self, user_id: str, user_events: pd.DataFrame, request_date: str) -> Dict[str, RecurringExpense]:
        req_d = str(request_date)[:10]
        hist = user_events[
            (user_events['status'] == 'settled')
            & (user_events['direction'] == 'debit')
            & (user_events['settlement_date'] < req_d)
        ].copy()

        commitments = {}
        for cat in self.ALL_RECURRING_CATEGORIES:
            c_events = hist[hist['category'] == cat].sort_values('settlement_date')
            if len(c_events) < 2:
                continue

            settle_dates = pd.to_datetime(c_events['settlement_date'])
            diffs = settle_dates.diff().dt.days.dropna()
            if diffs.empty:
                continue
            med_diff = float(diffs.median())

            amts = c_events['home_amount'].dropna()
            if amts.empty:
                continue
            amt = float(amts.median())
            flex = str(c_events['flexibility'].iloc[-1]) if 'flexibility' in c_events.columns else 'fixed'

            if user_id in self.rent_multipliers and cat == 'rent':
                amt *= self.rent_multipliers[user_id]

            # Weekly (5-9 days)
            if 5.0 <= med_diff <= 9.0:
                last_d = settle_dates.iloc[-1]
                commitments[cat] = RecurringExpense(category=cat, cadence='weekly', amount=amt, day_of_week=last_d.dayofweek, anchor_date=str(last_d)[:10], flexibility=flex)
            # Biweekly (12-16 days)
            elif 12.0 <= med_diff <= 16.0:
                last_d = settle_dates.iloc[-1]
                commitments[cat] = RecurringExpense(category=cat, cadence='biweekly', amount=amt, anchor_date=str(last_d)[:10], flexibility=flex)
            # Triweekly (19-23 days)
            elif 19.0 <= med_diff <= 23.0:
                last_d = settle_dates.iloc[-1]
                commitments[cat] = RecurringExpense(category=cat, cadence='triweekly', amount=amt, anchor_date=str(last_d)[:10], flexibility=flex)
            # Monthly (25-35 days)
            elif 25.0 <= med_diff <= 35.0:
                typ_day = int(settle_dates.dt.day.mode().iloc[0])
                commitments[cat] = RecurringExpense(category=cat, cadence='monthly', amount=amt, day_of_month=typ_day, flexibility=flex)
            elif len(c_events) >= 3 and settle_dates.dt.day.nunique() <= 2:
                typ_day = int(settle_dates.dt.day.mode().iloc[0])
                commitments[cat] = RecurringExpense(category=cat, cadence='monthly', amount=amt, day_of_month=typ_day, flexibility=flex)

        return commitments


# Test cashflow simulator with triweekly support
class EnhancedDailyCashFlowSimulator(DailyCashFlowSimulator):
    def simulate(self, user_id, user_events, current_available_balance, minimum_balance_to_keep, requested_amount, request_date):
        # Run standard simulation but ensure triweekly expenses are recursed
        req_d = datetime.strptime(str(request_date)[:10], "%Y-%m-%d")
        pending_total, _ = self.expense_forecaster.get_pending_debits_total(user_events, request_date)
        initial_balance = current_available_balance - pending_total

        sched_incomes = self.income_forecaster.get_future_scheduled_income(user_events, request_date)
        sched_income_map = {s_date: amt for s_date, amt in sched_incomes}

        sched_debits = self.expense_forecaster.get_future_scheduled_debits(user_events, request_date)
        sched_debit_map = {}
        scheduled_categories_by_month = set()
        for _, s_date, amt, cat in sched_debits:
            sched_debit_map[s_date] = sched_debit_map.get(s_date, 0.0) + amt
            sd_dt = datetime.strptime(s_date, "%Y-%m-%d")
            scheduled_categories_by_month.add((cat, sd_dt.year, sd_dt.month))

        recurring_commitments = self.expense_forecaster.get_recurring_commitments(user_id, user_events, request_date)
        salary_schedule = self.income_forecaster.get_recurring_salary_schedule(user_id, user_events, request_date)

        var_stats = self.expense_forecaster.get_variable_essential_stats(user_events, request_date)
        weekly_stat = var_stats.get(self.stat_key, var_stats.get('median', 0.0))
        daily_var_rate = weekly_stat / 7.0 if weekly_stat > 0 else 0.0

        current_bal = initial_balance
        min_balance = current_bal
        min_balance_date = req_d.strftime("%Y-%m-%d")
        next_income_date = None
        daily_trajectory = [(min_balance_date, current_bal)]

        for day_offset in range(self.horizon_days):
            cur_date = req_d + timedelta(days=day_offset)
            date_str = cur_date.strftime("%Y-%m-%d")

            daily_income = 0.0
            daily_expenses = 0.0

            if date_str in sched_income_map:
                daily_income += sched_income_map[date_str]
                if next_income_date is None and day_offset > 0:
                    next_income_date = date_str

            if salary_schedule is not None and date_str not in sched_income_map:
                is_pay_day = False
                if salary_schedule.cadence == 'monthly' and cur_date.day == salary_schedule.day_of_month:
                    is_pay_day = True
                elif salary_schedule.cadence == 'weekly' and cur_date.weekday() == salary_schedule.day_of_week:
                    is_pay_day = True
                elif salary_schedule.cadence == 'biweekly' and salary_schedule.anchor_date:
                    anchor_dt = datetime.strptime(salary_schedule.anchor_date, "%Y-%m-%d")
                    if (cur_date - anchor_dt).days > 0 and (cur_date - anchor_dt).days % 14 == 0:
                        is_pay_day = True
                if is_pay_day:
                    daily_income += salary_schedule.amount
                    if next_income_date is None and day_offset > 0:
                        next_income_date = date_str

            if date_str in sched_debit_map:
                daily_expenses += sched_debit_map[date_str]

            for cat, rec_exp in recurring_commitments.items():
                if (cat, cur_date.year, cur_date.month) in scheduled_categories_by_month:
                    continue

                is_expense_day = False
                if rec_exp.cadence == 'monthly' and cur_date.day == rec_exp.day_of_month:
                    is_expense_day = True
                elif rec_exp.cadence == 'weekly' and cur_date.weekday() == rec_exp.day_of_week:
                    is_expense_day = True
                elif rec_exp.cadence == 'biweekly' and rec_exp.anchor_date:
                    anchor_dt = datetime.strptime(rec_exp.anchor_date, "%Y-%m-%d")
                    if (cur_date - anchor_dt).days > 0 and (cur_date - anchor_dt).days % 14 == 0:
                        is_expense_day = True
                elif rec_exp.cadence == 'triweekly' and rec_exp.anchor_date:
                    anchor_dt = datetime.strptime(rec_exp.anchor_date, "%Y-%m-%d")
                    if (cur_date - anchor_dt).days > 0 and (cur_date - anchor_dt).days % 21 == 0:
                        is_expense_day = True

                if is_expense_day:
                    daily_expenses += rec_exp.amount

            daily_expenses += daily_var_rate
            current_bal = current_bal + daily_income - daily_expenses
            daily_trajectory.append((date_str, current_bal))

            if current_bal < min_balance:
                min_balance = current_bal
                min_balance_date = date_str

        safe_amount = max(0.0, min(requested_amount, min_balance - minimum_balance_to_keep))

        from code.forecasting.simulator import SimulationResult
        return SimulationResult(
            user_id=user_id,
            request_date=str(request_date)[:10],
            current_balance=current_available_balance,
            minimum_balance=minimum_balance_to_keep,
            requested_amount=requested_amount,
            next_confirmed_income_date=next_income_date,
            projected_minimum_balance_date=min_balance_date,
            projected_minimum_balance=min_balance,
            predicted_reserve=max(0.0, current_available_balance - min_balance),
            amount_safe_to_pay=safe_amount,
            daily_balances=daily_trajectory,
        )

# Evaluate on target requests
inc_f_enh = EnhancedIncomeForecaster(messages)
exp_f_enh = EnhancedExpenseForecaster()
sim_enh = EnhancedDailyCashFlowSimulator(inc_f_enh, exp_f_enh, 'median', 90)

print("=" * 95)
print(f"{'Req ID':<10} | {'User':<8} | {'GT Safe':<12} | {'Old Safe':<12} | {'New Safe':<12} | {'New Min Date':<12}")
print("-" * 95)

profiles_map = {str(p['user_id']).strip(): p for _, p in profiles.iterrows()}

for _, row in sample_requests.iterrows():
    r_id = str(row['request_id']).strip()
    u_id = str(row['user_id']).strip()
    req_d = str(row['request_date'])[:10]
    req_amt = float(row['requested_amount'])
    p = profiles_map[u_id]
    curr_b = float(p['current_available_balance'])
    min_b = float(p['minimum_balance_to_keep'])
    home_c = p['home_currency']

    u_ev = norm.get_user_events(u_id, home_c, req_d)
    res_enh = sim_enh.simulate(u_id, u_ev, curr_b, min_b, req_amt, req_d)

    gt_safe = float(row['amount_safe_to_pay'])
    print(f"{r_id:<10} | {u_id:<8} | {gt_safe:<12.2f} | {res_enh.amount_safe_to_pay:<12.2f} | {res_enh.projected_minimum_balance_date:<12}")
