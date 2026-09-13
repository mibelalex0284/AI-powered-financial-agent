import os
import sys
import re
import numpy as np
import pandas as pd
from datetime import datetime, timedelta

REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

from code.forecasting.normalization import EventNormalizer, ExchangeRateProvider, ImageAmountResolver
from code.forecasting.income import ConfirmedIncomeForecaster, RecurringIncomeSchedule
from code.forecasting.expenses import ExpenseForecaster, RecurringExpense
from code.forecasting.simulator import DailyCashFlowSimulator, SimulationResult
from code.optimization.payment_optimizer import Payment, CandidatePlan, PlanSafetyEvaluator, UserBaselineCashflows
from code.optimization.spending_changes import SpendingChangeOptimizer
from code.optimization.decision_engine import DecisionEngine


class EnhancedIncomeForecaster(ConfirmedIncomeForecaster):
    def _parse_message_rules(self):
        super()._parse_message_rules()
        self.pay_day_overrides = {}
        if self.messages_df.empty:
            return

        for _, m in self.messages_df.iterrows():
            uid = str(m.get('user_id', '')).strip()
            txt = str(m.get('message_text', ''))
            # General authoritative payroll date replacement
            if 'replaces the payroll date' in txt.lower() or 'menggantikan tanggal penggajian' in txt.lower():
                dates = re.findall(r'\d{4}-\d{2}-\d{2}', txt)
                if dates:
                    d_obj = datetime.strptime(dates[0], "%Y-%m-%d")
                    self.pay_day_overrides[uid] = d_obj.day

    def get_recurring_salary_schedule(self, user_id, user_events, request_date):
        if not self.has_future_confirmed_income(user_id, user_events):
            return None

        override_amt = self.salary_overrides.get(user_id)
        override_day = self.pay_day_overrides.get(user_id)
        req_d = str(request_date)[:10]

        sched_sal = user_events[
            (user_events['category'] == 'salary')
            & (user_events['status'] == 'scheduled')
            & (user_events['direction'] == 'credit')
            & (user_events['settlement_date'] >= req_d)
        ]

        hist_sal = user_events[
            (user_events['category'] == 'salary')
            & (user_events['status'] == 'settled')
            & (user_events['direction'] == 'credit')
            & (user_events['settlement_date'] < req_d)
        ].copy()

        if not hist_sal.empty:
            non_base_mask = hist_sal['description'].str.lower().str.contains('bonus|commission|second household|arrears')
            base_sal = hist_sal[~non_base_mask]
            if not base_sal.empty:
                hist_sal = base_sal

        if hist_sal.empty:
            if not sched_sal.empty:
                s_row = sched_sal.iloc[0]
                s_date = datetime.strptime(str(s_row['settlement_date'])[:10], "%Y-%m-%d")
                amt = override_amt if override_amt is not None else float(s_row['home_amount'])
                day = override_day if override_day is not None else s_date.day
                return RecurringIncomeSchedule(cadence='monthly', amount=amt, day_of_month=day)
            return None

        hist_sal = hist_sal.sort_values('settlement_date')
        settle_dates = pd.to_datetime(hist_sal['settlement_date'])
        diffs = settle_dates.diff().dt.days.dropna()
        median_interval = float(diffs.median()) if not diffs.empty else 30.0

        # Amount determination: scheduled confirmed salary takes precedence over prorated historical
        if override_amt is not None:
            amt = override_amt
        elif not sched_sal.empty and any('prorated' in str(d).lower() for d in hist_sal['description']):
            amt = float(sched_sal.iloc[0]['home_amount'])
        else:
            amts = hist_sal['home_amount'].dropna()
            amt = float(amts.median()) if not amts.empty else 0.0

        # Weekly cadence (5 to 9 days)
        if 5.0 <= median_interval <= 9.0:
            last_date = settle_dates.iloc[-1]
            return RecurringIncomeSchedule(cadence='weekly', amount=amt, day_of_week=last_date.dayofweek)

        # Biweekly cadence (12 to 16 days)
        if 12.0 <= median_interval <= 16.0:
            last_date = settle_dates.iloc[-1]
            return RecurringIncomeSchedule(cadence='biweekly', amount=amt, anchor_date=str(last_date)[:10])

        # Monthly cadence
        # Precedence for pay day:
        # 1. Authoritative message date override
        # 2. Scheduled salary day
        # 3. Mode of historical pay days
        if override_day is not None:
            day = override_day
        elif not sched_sal.empty:
            s_date = datetime.strptime(str(sched_sal.iloc[0]['settlement_date'])[:10], "%Y-%m-%d")
            day = s_date.day
        else:
            day = int(settle_dates.dt.day.mode().iloc[0])

        return RecurringIncomeSchedule(cadence='monthly', amount=amt, day_of_month=day)


class EnhancedExpenseForecaster(ExpenseForecaster):
    FIXED_CATEGORIES = [
        'rent', 'utilities', 'debt_repayment', 'education', 'cloud_storage',
        'streaming', 'music_subscription', 'delivery_membership', 'gym',
        'insurance', 'family_support', 'housing', 'healthcare',
        'dining', 'entertainment', 'shopping', 'investment'
    ]

    def get_recurring_commitments(self, user_id, user_events, request_date):
        req_d = str(request_date)[:10]
        hist = user_events[
            (user_events['status'] == 'settled')
            & (user_events['direction'] == 'debit')
            & (user_events['settlement_date'] < req_d)
        ].copy()

        commitments = {}
        for cat in self.FIXED_CATEGORIES:
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

            # Weekly cadence (5-9 days)
            if 5.0 <= med_diff <= 9.0:
                last_d = settle_dates.iloc[-1]
                commitments[cat] = RecurringExpense(
                    category=cat, cadence='weekly', amount=amt, day_of_week=last_d.dayofweek, anchor_date=str(last_d)[:10], flexibility=flex
                )
            # Biweekly cadence (12-16 days)
            elif 12.0 <= med_diff <= 16.0:
                last_d = settle_dates.iloc[-1]
                commitments[cat] = RecurringExpense(
                    category=cat, cadence='biweekly', amount=amt, anchor_date=str(last_d)[:10], flexibility=flex
                )
            # Tri-weekly cadence (19-23 days)
            elif 19.0 <= med_diff <= 23.0:
                last_d = settle_dates.iloc[-1]
                commitments[cat] = RecurringExpense(
                    category=cat, cadence='triweekly', amount=amt, anchor_date=str(last_d)[:10], flexibility=flex
                )
            # Monthly cadence (25-35 days)
            elif 25.0 <= med_diff <= 35.0:
                typ_day = int(settle_dates.dt.day.mode().iloc[0])
                commitments[cat] = RecurringExpense(
                    category=cat, cadence='monthly', amount=amt, day_of_month=typ_day, flexibility=flex
                )
            elif len(c_events) >= 3 and settle_dates.dt.day.nunique() <= 2:
                typ_day = int(settle_dates.dt.day.mode().iloc[0])
                commitments[cat] = RecurringExpense(
                    category=cat, cadence='monthly', amount=amt, day_of_month=typ_day, flexibility=flex
                )
        return commitments


class EnhancedDailyCashFlowSimulator(DailyCashFlowSimulator):
    def simulate(self, user_id, user_events, current_available_balance, minimum_balance_to_keep, requested_amount, request_date):
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

        predicted_reserve = max(0.0, current_available_balance - min_balance)
        safe_amount = max(0.0, min(requested_amount, min_balance - minimum_balance_to_keep))

        return SimulationResult(
            user_id=user_id,
            request_date=str(request_date)[:10],
            current_balance=current_available_balance,
            minimum_balance=minimum_balance_to_keep,
            requested_amount=requested_amount,
            next_confirmed_income_date=next_income_date,
            projected_minimum_balance_date=min_balance_date,
            projected_minimum_balance=min_balance,
            predicted_reserve=predicted_reserve,
            amount_safe_to_pay=safe_amount,
            daily_balances=daily_trajectory,
        )


class EnhancedPlanSafetyEvaluator(PlanSafetyEvaluator):
    def is_plan_safe(self, baseline, payments, spending_changes=None, horizon_days=90):
        req_d = datetime.strptime(str(baseline.request_date)[:10], "%Y-%m-%d")

        stopped_categories = set()
        reduced_categories = {}
        if spending_changes and spending_changes.changes:
            for sc in spending_changes.changes:
                if sc.action == 'stop':
                    stopped_categories.add(sc.category)
                elif sc.action == 'reduce_to':
                    reduced_categories[sc.category] = sc.new_amount

        payment_map = {}
        for p in payments:
            payment_map[p.date] = payment_map.get(p.date, 0.0) + p.amount

        current_bal = baseline.initial_balance
        min_balance = current_bal
        min_balance_date = req_d.strftime("%Y-%m-%d")

        sched_inc = baseline.sched_income_map
        sched_deb = baseline.sched_debit_map
        sched_cats_month = baseline.scheduled_categories_by_month
        sal = baseline.salary_schedule
        recs = baseline.recurring_commitments
        daily_var = baseline.daily_var_rate

        for day_offset in range(horizon_days):
            cur_date = req_d + timedelta(days=day_offset)
            date_str = cur_date.strftime("%Y-%m-%d")

            daily_income = 0.0
            daily_expenses = 0.0

            if date_str in sched_inc:
                daily_income += sched_inc[date_str]
            elif sal is not None:
                is_pay = False
                if sal.cadence == 'monthly' and cur_date.day == sal.day_of_month:
                    is_pay = True
                elif sal.cadence == 'weekly' and cur_date.weekday() == sal.day_of_week:
                    is_pay = True
                elif sal.cadence == 'biweekly' and sal.anchor_date:
                    anchor_dt = datetime.strptime(sal.anchor_date, "%Y-%m-%d")
                    if (cur_date - anchor_dt).days > 0 and (cur_date - anchor_dt).days % 14 == 0:
                        is_pay = True
                if is_pay:
                    daily_income += sal.amount

            if date_str in sched_deb:
                daily_expenses += sched_deb[date_str]

            for rc in recs:
                if (rc.category, cur_date.year, cur_date.month) in sched_cats_month:
                    continue
                if rc.category in stopped_categories:
                    continue

                rc_amt = reduced_categories.get(rc.category, rc.amount)

                is_due = False
                if rc.cadence == 'monthly' and cur_date.day == rc.day_of_month:
                    is_due = True
                elif rc.cadence == 'weekly' and cur_date.weekday() == rc.day_of_week:
                    is_due = True
                elif rc.cadence == 'biweekly' and rc.anchor_date:
                    anchor_dt = datetime.strptime(rc.anchor_date, "%Y-%m-%d")
                    if (cur_date - anchor_dt).days > 0 and (cur_date - anchor_dt).days % 14 == 0:
                        is_due = True
                elif rc.cadence == 'triweekly' and rc.anchor_date:
                    anchor_dt = datetime.strptime(rc.anchor_date, "%Y-%m-%d")
                    if (cur_date - anchor_dt).days > 0 and (cur_date - anchor_dt).days % 21 == 0:
                        is_due = True

                if is_due:
                    daily_expenses += rc_amt

            daily_expenses += daily_var
            if date_str in payment_map:
                daily_expenses += payment_map[date_str]

            current_bal = current_bal + daily_income - daily_expenses
            if current_bal < min_balance:
                min_balance = current_bal
                min_balance_date = date_str

        is_safe = (min_balance >= baseline.minimum_balance_to_keep - 1e-4)
        return is_safe, min_balance, min_balance_date


# Evaluate across all 25 samples
def run_evaluation():
    dataset_dir = 'dataset'
    profiles_df = pd.read_csv(os.path.join(dataset_dir, 'financial_profiles.csv'))
    events_df = pd.read_csv(os.path.join(dataset_dir, 'financial_events.csv'))
    rates_df = pd.read_csv(os.path.join(dataset_dir, 'exchange_rates.csv'))
    sample_df = pd.read_csv(os.path.join(dataset_dir, 'sample_requests.csv'))
    options_df = pd.read_csv(os.path.join(dataset_dir, 'request_payment_options.csv'))
    messages_df = pd.read_csv(os.path.join(dataset_dir, 'messages.csv'))
    images_df = pd.read_csv(os.path.join(dataset_dir, 'images.csv'))

    normalizer = EventNormalizer(events_df, images_df, ExchangeRateProvider(rates_df))
    income_forecaster = EnhancedIncomeForecaster(messages_df)
    expense_forecaster = EnhancedExpenseForecaster()

    engine = DecisionEngine(normalizer, income_forecaster, expense_forecaster, variable_spending_stat='median')
    engine.simulator = EnhancedDailyCashFlowSimulator(income_forecaster, expense_forecaster, variable_spending_stat='median', forecast_horizon_days=90)
    engine.evaluator = EnhancedPlanSafetyEvaluator(income_forecaster, expense_forecaster, variable_spending_stat='median')

    profile_map = {str(p['user_id']).strip(): p for _, p in profiles_df.iterrows()}

    status_matches = 0
    method_matches = 0
    plan_matches = 0
    date_matches = 0
    change_matches = 0
    exact_safe = 0
    abs_errors = []
    norm_errors = []

    print(f"{'Req ID':<10} | {'Pred Safe':<12} | {'GT Safe':<12} | {'Abs Err':<10} | {'Status (P/GT)':<27} | {'Method (P/GT)':<24} | {'Plan':<5} | {'Earliest (P/GT)':<23} | {'Spending Changes (P/GT)'}")
    print("-" * 150)

    for _, req in sample_df.iterrows():
        rid = str(req['request_id']).strip()
        uid = str(req['user_id']).strip()
        user_prof = profile_map[uid]
        p_opts = options_df[options_df['request_id'] == rid]

        res = engine.evaluate_request(req, user_prof, p_opts)

        gt_safe = float(req['amount_safe_to_pay'])
        gt_status = str(req['affordability_status']).strip()
        gt_method = str(req['recommended_payment_method']).strip()
        gt_plan = str(req.get('payment_plan', 'none')).strip()
        gt_date = str(req.get('earliest_date_for_full_payment', '')).strip()
        if gt_date == 'nan':
            gt_date = ''
        gt_changes = str(req.get('spending_changes_needed', 'none')).strip()

        p_safe = round(res.amount_safe_to_pay, 2)
        p_status = res.affordability_status
        p_method = res.recommended_payment_method
        p_plan = res.payment_plan
        p_date = res.earliest_date_for_full_payment or ''
        p_changes = res.spending_changes_needed

        err = abs(p_safe - gt_safe)
        abs_errors.append(err)
        req_amt = float(req['requested_amount'])
        norm_err = (err / req_amt) * 100.0 if req_amt > 0 else 0.0
        norm_errors.append(norm_err)

        if err < 0.05:
            exact_safe += 1

        s_ok = (p_status == gt_status)
        m_ok = (p_method == gt_method)
        p_ok = (p_plan == gt_plan)
        d_ok = (p_date == gt_date)
        c_ok = (p_changes == gt_changes)

        if s_ok: status_matches += 1
        if m_ok: method_matches += 1
        if p_ok: plan_matches += 1
        if d_ok: date_matches += 1
        if c_ok: change_matches += 1

        plan_flag = 'OK' if p_ok else 'DIFF'
        print(f"{rid:<10} | {p_safe:<12.2f} | {gt_safe:<12.2f} | {err:<10.2f} | {p_status[:12]:<12} / {gt_status[:12]:<12} | {p_method[:11]:<11} / {gt_method[:11]:<11} | {plan_flag:<5} | {p_date[:10]:<10} / {gt_date[:10]:<10} | {p_changes} / {gt_changes}")

    n = len(sample_df)
    print("=" * 150)
    print("SUMMARY RESULTS (25 SAMPLES):")
    print(f"  Exact Safe Matches:       {exact_safe} / {n} ({exact_safe/n*100:.1f}%)")
    print(f"  Safe MAE:                 {np.mean(abs_errors):,.2f}")
    print(f"  Normalized Error:         {np.mean(norm_errors):.2f}%")
    print(f"  Status Accuracy:          {status_matches} / {n} ({status_matches/n*100:.1f}%)")
    print(f"  Payment Method Accuracy:  {method_matches} / {n} ({method_matches/n*100:.1f}%)")
    print(f"  Payment Plan Accuracy:    {plan_matches} / {n} ({plan_matches/n*100:.1f}%)")
    print(f"  Earliest Date Accuracy:   {date_matches} / {n} ({date_matches/n*100:.1f}%)")
    print(f"  Spending Changes Accuracy:{change_matches} / {n} ({change_matches/n*100:.1f}%)")
    print("=" * 150)

if __name__ == '__main__':
    run_evaluation()
