import sys
import os
sys.path.insert(0, os.path.abspath('.'))
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import List, Optional, Tuple, Set, Dict

from code.forecasting import (
    EventNormalizer,
    ImageAmountResolver,
    ExchangeRateProvider,
    ConfirmedIncomeForecaster,
    ExpenseForecaster,
    DailyCashFlowSimulator,
)
from code.optimization.spending_changes import SpendingChangeOptimizer, SpendingChangePlan
from code.optimization.payment_optimizer import Payment, CandidatePlan, PlanSafetyEvaluator

# Load datasets
profiles = pd.read_csv('dataset/financial_profiles.csv')
events = pd.read_csv('dataset/financial_events.csv')
sample_requests = pd.read_csv('dataset/sample_requests.csv')
options = pd.read_csv('dataset/request_payment_options.csv')
messages = pd.read_csv('dataset/messages.csv')
rates = pd.read_csv('dataset/exchange_rates.csv')
images = pd.read_csv('dataset/images.csv')

fx = ExchangeRateProvider(rates)
norm = EventNormalizer(events, images, fx)
inc_f = ConfirmedIncomeForecaster(messages)
exp_f = ExpenseForecaster()
evaluator = PlanSafetyEvaluator(inc_f, exp_f, variable_spending_stat='median')
sp_opt = SpendingChangeOptimizer()
sim = DailyCashFlowSimulator(inc_f, exp_f, variable_spending_stat='median', forecast_horizon_days=90)

def find_earliest_full_payment_date(baseline, req_amt, req_d_str, evaluator):
    """
    Find the earliest conservative projected date for one safe full payment.
    Must keep balance >= minimum_balance_to_keep throughout the 90 days.
    """
    req_d = datetime.strptime(req_d_str, "%Y-%m-%d")

    # If safe today, return request_date
    pay_today = [Payment(date=req_d_str, amount=req_amt)]
    is_safe_today, _, _ = evaluator.is_plan_safe(baseline, pay_today, spending_changes=None, horizon_days=90)
    if is_safe_today:
        return req_d_str

    # Search future dates: a single full payment of req_amt on candidate_date
    # For a candidate date t, after paying req_amt on date t, does the balance
    # trajectory from day 0 to 90 remain >= minimum_balance_to_keep?
    # Note: payment on day t only affects balances from day t onwards.
    # To be conservative, we only consider dates where the payment leaves balance >= min_balance
    # AND subsequent essential expenses until next paycheck (or horizon) are covered.
    for d_off in range(1, 90):
        test_d = (req_d + timedelta(days=d_off)).strftime("%Y-%m-%d")
        pay_test = [Payment(date=test_d, amount=req_amt)]
        is_safe, min_bal_reached, _ = evaluator.is_plan_safe(baseline, pay_test, spending_changes=None, horizon_days=90)
        if is_safe:
            # Also verify that balance immediately after payment on test_d is >= minimum_balance
            # and that it's not a boundary artifact
            if d_off < 85:  # Avoid horizon boundary artifacts where future expenses aren't simulated
                return test_d

    return None

results = []
for idx, req in sample_requests.iterrows():
    r_id = req['request_id']
    u_id = req['user_id']
    p = profiles[profiles['user_id'] == u_id].iloc[0]
    home_curr = str(p['home_currency'])
    curr_bal = float(p['current_available_balance'])
    min_bal = float(p['minimum_balance_to_keep'])
    req_amt = float(req['requested_amount'])
    req_d_str = str(req['request_date'])[:10]
    comp_d_str = str(req['desired_completion_date'])[:10]
    allows_partial = bool(req['allows_partial_payment'])

    user_methods = set(m.strip() for m in str(p.get('payment_methods_user_will_consider', '')).split('|'))
    max_inst_months = float(p['max_installment_months']) if pd.notna(p.get('max_installment_months')) else 0.0

    u_events = norm.get_user_events(u_id, home_curr, req_d_str)
    baseline = evaluator.build_baseline_cashflows(u_id, u_events, curr_bal, min_bal, req_d_str)

    # 1. amount_safe_to_pay today
    sim_res = sim.simulate(u_id, u_events, curr_bal, min_bal, req_amt, req_d_str)
    safe_today = sim_res.amount_safe_to_pay

    # 2. earliest_date_for_full_payment
    earliest_date = find_earliest_full_payment_date(baseline, req_amt, req_d_str, evaluator)

    # 3. Candidate plans
    candidate_plans: List[CandidatePlan] = []
    spending_plans = sp_opt.get_candidate_plans(u_id, p, u_events, req_d_str)

    # Candidate A: full_payment today
    if 'full_payment' in user_methods:
        for sp in spending_plans:
            # If 0 spending changes, only allowed if safe_today >= req_amt
            if sp.count == 0 and safe_today < req_amt:
                continue

            pay_full = [Payment(date=req_d_str, amount=req_amt)]
            is_s, _, _ = evaluator.is_plan_safe(baseline, pay_full, spending_changes=sp, horizon_days=90)
            if is_s:
                status = 'affordable_now' if sp.count == 0 else 'affordable_with_plan'
                candidate_plans.append(CandidatePlan(
                    payment_method='full_payment',
                    payment_option_id='option_full',
                    payments=pay_full,
                    spending_changes=sp,
                    total_payable_amount=req_amt,
                    start_date=req_d_str,
                    completion_date=req_d_str,
                    number_of_payments=1,
                    is_safe=True,
                    completes_by_deadline=(req_d_str <= comp_d_str),
                    affordability_status=status,
                    earliest_date_for_full_payment=earliest_date or req_d_str
                ))

    # Candidate B: installments
    if 'installments' in user_methods and max_inst_months > 0:
        req_opts = options[options['request_id'] == r_id]
        inst_opts = req_opts[req_opts['payment_method'] == 'installments']
        for _, opt in inst_opts.iterrows():
            n_payments = int(opt['number_of_payments'])
            if n_payments > max_inst_months:
                continue

            p_amt = float(opt['payment_amount'])
            freq_days = float(opt['payment_frequency_days']) if pd.notna(opt['payment_frequency_days']) else 30.0
            start_d_str = str(opt['first_payment_date'])[:10]
            start_dt = datetime.strptime(start_d_str, "%Y-%m-%d")

            payments = []
            for i in range(n_payments):
                p_dt = start_dt + timedelta(days=int(i * freq_days))
                payments.append(Payment(date=p_dt.strftime("%Y-%m-%d"), amount=p_amt))

            comp_d_plan = payments[-1].date
            completes_on_time = (comp_d_plan <= comp_d_str)
            total_cost = float(opt['total_payable_amount']) if pd.notna(opt.get('total_payable_amount')) else sum(p.amount for p in payments)

            for sp in spending_plans:
                is_s, _, _ = evaluator.is_plan_safe(baseline, payments, spending_changes=sp, horizon_days=90)
                if is_s:
                    candidate_plans.append(CandidatePlan(
                        payment_method='installments',
                        payment_option_id=str(opt['payment_option_id']),
                        payments=payments,
                        spending_changes=sp,
                        total_payable_amount=total_cost,
                        start_date=start_d_str,
                        completion_date=comp_d_plan,
                        number_of_payments=n_payments,
                        is_safe=True,
                        completes_by_deadline=completes_on_time,
                        affordability_status='affordable_with_plan',
                        earliest_date_for_full_payment=earliest_date
                    ))

    # Candidate C: partial_payment
    if allows_partial and ('partial_payment' in user_methods):
        if 0 < safe_today < req_amt and earliest_date is not None:
            if earliest_date <= comp_d_str:
                p_partial = [
                    Payment(date=req_d_str, amount=safe_today),
                    Payment(date=earliest_date, amount=req_amt - safe_today)
                ]
                for sp in spending_plans:
                    is_s, _, _ = evaluator.is_plan_safe(baseline, p_partial, spending_changes=sp, horizon_days=90)
                    if is_s:
                        candidate_plans.append(CandidatePlan(
                            payment_method='partial_payment',
                            payment_option_id='option_partial',
                            payments=p_partial,
                            spending_changes=sp,
                            total_payable_amount=req_amt,
                            start_date=req_d_str,
                            completion_date=earliest_date,
                            number_of_payments=2,
                            is_safe=True,
                            completes_by_deadline=True,
                            affordability_status='affordable_with_plan',
                            earliest_date_for_full_payment=earliest_date
                        ))

    # Candidate D: wait (completes by earliest_date)
    if earliest_date is not None and ('full_payment' in user_methods):
        completes_on_time = (earliest_date <= comp_d_str)
        p_wait = [Payment(date=earliest_date, amount=req_amt)]
        candidate_plans.append(CandidatePlan(
            payment_method='wait',
            payment_option_id='option_wait',
            payments=p_wait,
            spending_changes=SpendingChangePlan(changes=[]),
            total_payable_amount=req_amt,
            start_date=earliest_date,
            completion_date=earliest_date,
            number_of_payments=1,
            is_safe=True,
            completes_by_deadline=completes_on_time,
            affordability_status='affordable_later',
            earliest_date_for_full_payment=earliest_date
        ))

    # Selection logic:
    # 1. Look for plans completing by desired_completion_date
    on_time_plans = [p for p in candidate_plans if p.completes_by_deadline]

    chosen_plan = None
    if on_time_plans:
        # Ranking rules:
        # 1. No spending changes first (count == 0)
        # 2. Minimize total payable amount
        # 3. Start payment earlier
        # 4. Fewer payments
        # 5. Lowest payment_option_id
        on_time_plans.sort(key=lambda p: (
            p.spending_changes.count,
            p.total_payable_amount,
            p.start_date,
            p.number_of_payments,
            p.payment_option_id
        ))
        chosen_plan = on_time_plans[0]
    elif earliest_date is not None and ('full_payment' in user_methods):
        # Fallback to wait if full payment is safe later (even if after completion date)
        chosen_plan = CandidatePlan(
            payment_method='wait',
            payment_option_id='option_wait',
            payments=[Payment(date=earliest_date, amount=req_amt)],
            spending_changes=SpendingChangePlan(changes=[]),
            total_payable_amount=req_amt,
            start_date=earliest_date,
            completion_date=earliest_date,
            number_of_payments=1,
            is_safe=True,
            completes_by_deadline=False,
            affordability_status='affordable_later',
            earliest_date_for_full_payment=earliest_date
        )
    else:
        # Fallback: not_recommended
        chosen_plan = CandidatePlan(
            payment_method='not_recommended',
            payment_option_id='none',
            payments=[],
            spending_changes=SpendingChangePlan(changes=[]),
            total_payable_amount=0.0,
            start_date="",
            completion_date="",
            number_of_payments=0,
            is_safe=False,
            completes_by_deadline=False,
            affordability_status='not_affordable',
            earliest_date_for_full_payment=None
        )

    results.append({
        'request_id': r_id,
        'user_id': u_id,
        'pred_safe': safe_today,
        'gt_safe': float(req['amount_safe_to_pay']),
        'pred_status': chosen_plan.affordability_status,
        'gt_status': req['affordability_status'],
        'pred_method': chosen_plan.payment_method,
        'gt_method': req['recommended_payment_method'],
        'pred_plan': chosen_plan.plan_string,
        'gt_plan': req['payment_plan'],
        'pred_earliest': chosen_plan.earliest_date_for_full_payment if chosen_plan.affordability_status != 'not_affordable' else "",
        'gt_earliest': str(req['earliest_date_for_full_payment']) if pd.notna(req.get('earliest_date_for_full_payment')) else "",
        'pred_changes': chosen_plan.spending_changes.spec_string,
        'gt_changes': req['spending_changes_needed']
    })

df_eval = pd.DataFrame(results)
print("\n===========================================================")
print("REFINED ACCURACY METRICS ACROSS 25 SAMPLE REQUESTS")
print("===========================================================")
print(f"Status Match:          {(df_eval['pred_status'] == df_eval['gt_status']).sum()} / 25  ({(df_eval['pred_status'] == df_eval['gt_status']).mean()*100:.1f}%)")
print(f"Method Match:          {(df_eval['pred_method'] == df_eval['gt_method']).sum()} / 25  ({(df_eval['pred_method'] == df_eval['gt_method']).mean()*100:.1f}%)")
print(f"Plan Match:            {(df_eval['pred_plan'] == df_eval['gt_plan']).sum()} / 25  ({(df_eval['pred_plan'] == df_eval['gt_plan']).mean()*100:.1f}%)")
print(f"Spending Changes Match:{(df_eval['pred_changes'] == df_eval['gt_changes']).sum()} / 25  ({(df_eval['pred_changes'] == df_eval['gt_changes']).mean()*100:.1f}%)")
print(f"Earliest Date Match:   {(df_eval['pred_earliest'] == df_eval['gt_earliest']).sum()} / 25  ({(df_eval['pred_earliest'] == df_eval['gt_earliest']).mean()*100:.1f}%)")

print("\nDETAILED PER-REQUEST COMPARISON:")
for idx, r in df_eval.iterrows():
    s_ok = "OK" if r['pred_status'] == r['gt_status'] else "MIS"
    m_ok = "OK" if r['pred_method'] == r['gt_method'] else "MIS"
    p_ok = "OK" if r['pred_plan'] == r['gt_plan'] else "MIS"
    c_ok = "OK" if r['pred_changes'] == r['gt_changes'] else "MIS"
    e_ok = "OK" if r['pred_earliest'] == r['gt_earliest'] else "MIS"
    print(f"{r['request_id']} | Status: {s_ok} ({r['pred_status']}) | Method: {m_ok} ({r['pred_method']}) | Plan: {p_ok} | Changes: {c_ok} | Earliest: {e_ok}")
    if s_ok == "MIS" or m_ok == "MIS" or p_ok == "MIS" or c_ok == "MIS" or e_ok == "MIS":
        print(f"    GT:   status={r['gt_status']}, method={r['gt_method']}, plan={r['gt_plan']}, changes={r['gt_changes']}, earliest={r['gt_earliest']}")
        print(f"    PRED: status={r['pred_status']}, method={r['pred_method']}, plan={r['pred_plan']}, changes={r['pred_changes']}, earliest={r['pred_earliest']}")
