import sys
import os
sys.path.insert(0, os.path.abspath('.'))
import pandas as pd
import numpy as np
from datetime import datetime, timedelta

from code.forecasting import (
    EventNormalizer,
    ImageAmountResolver,
    ExchangeRateProvider,
    ConfirmedIncomeForecaster,
    ExpenseForecaster,
    DailyCashFlowSimulator,
)

# Load data
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

def run_evaluation(horizon_mode='90_day'):
    results = []
    
    for idx, req in sample_requests.iterrows():
        req_id = req['request_id']
        u_id = req['user_id']
        p = profiles[profiles['user_id'] == u_id].iloc[0]
        home_curr = str(p['home_currency'])
        req_d_str = str(req['request_date'])[:10]
        comp_d_str = str(req['desired_completion_date'])[:10]
        req_d = datetime.strptime(req_d_str, '%Y-%m-%d')
        comp_d = datetime.strptime(comp_d_str, '%Y-%m-%d')
        days_to_comp = max(1, (comp_d - req_d).days)
        
        u_events = norm.get_user_events(u_id, home_curr, req_d_str)
        curr_bal = float(p['current_available_balance'])
        min_bal = float(p['minimum_balance_to_keep'])
        req_amt = float(req['requested_amount'])
        gt_safe = float(req['amount_safe_to_pay'])
        
        # Determine horizon for this run
        if horizon_mode == 'A_full_90':
            h_days = 90
        elif horizon_mode == 'B_completion_date':
            h_days = days_to_comp
        elif horizon_mode == 'C_hybrid_90_completion':
            # 90-day safety horizon, but if salary exists, horizon is constrained to completion date or 90
            # Let's test standard 90 days with completion check
            h_days = 90
        elif horizon_mode == 'D_min_next_pay_or_90':
            # Next confirmed income or 90 days if no income
            sal = inc_f.get_recurring_salary_schedule(u_id, u_events, req_d_str)
            if sal is not None and sal.cadence == 'monthly':
                # days to next payday
                if sal.day_of_month >= req_d.day:
                    days_to_pay = sal.day_of_month - req_d.day
                else:
                    # next month
                    next_m = req_d.month + 1 if req_d.month < 12 else 1
                    next_y = req_d.year if req_d.month < 12 else req_d.year + 1
                    next_pay_dt = datetime(next_y, next_m, sal.day_of_month)
                    days_to_pay = (next_pay_dt - req_d).days
                h_days = max(1, days_to_pay)
            else:
                h_days = 90
        else:
            h_days = 90

        sim = DailyCashFlowSimulator(inc_f, exp_f, variable_spending_stat='median', forecast_horizon_days=h_days)
        sim_res = sim.simulate(
            user_id=u_id,
            user_events=u_events,
            current_available_balance=curr_bal,
            minimum_balance_to_keep=min_bal,
            requested_amount=req_amt,
            request_date=req_d_str
        )
        
        pred_safe = sim_res.amount_safe_to_pay
        err = abs(pred_safe - gt_safe)
        norm_err = (err / req_amt) * 100.0 if req_amt > 0 else 0.0
        
        results.append({
            'request_id': req_id,
            'user_id': u_id,
            'horizon_days': h_days,
            'gt_safe': gt_safe,
            'pred_safe': pred_safe,
            'error': err,
            'norm_error_pct': norm_err,
            'projected_min': sim_res.projected_minimum_balance,
            'proj_min_date': sim_res.projected_minimum_balance_date
        })
        
    df_res = pd.DataFrame(results)
    mae = df_res['error'].mean()
    rmse = np.sqrt((df_res['error']**2).mean())
    corr = df_res['pred_safe'].corr(df_res['gt_safe'])
    mean_norm_err = df_res['norm_error_pct'].mean()
    max_err = df_res['error'].max()
    
    return {
        'mode': horizon_mode,
        'mae': mae,
        'rmse': rmse,
        'pearson_r': corr,
        'mean_norm_err_pct': mean_norm_err,
        'max_error': max_err,
        'details': df_res
    }

modes = ['A_full_90', 'B_completion_date', 'D_min_next_pay_or_90']
eval_results = {}
for m in modes:
    eval_results[m] = run_evaluation(m)
    print(f"\n==================== MODE: {m} ====================")
    print(f"MAE: {eval_results[m]['mae']:.2f}")
    print(f"RMSE: {eval_results[m]['rmse']:.2f}")
    print(f"Pearson r: {eval_results[m]['pearson_r']:.4f}")
    print(f"Mean Norm Error %: {eval_results[m]['mean_norm_err_pct']:.2f}%")
    print(f"Max Error: {eval_results[m]['max_error']:.2f}")

# Print comparative table for key requests
print("\n==================== PER-REQUEST COMPARISON ====================")
r_ids = ['request_05', 'request_10', 'request_12', 'request_23']
for rid in r_ids:
    row_gt = sample_requests[sample_requests['request_id'] == rid].iloc[0]['amount_safe_to_pay']
    vals = [f"GT: {row_gt:.2f}"]
    for m in modes:
        d = eval_results[m]['details']
        p_val = d[d['request_id'] == rid].iloc[0]['pred_safe']
        h_d = d[d['request_id'] == rid].iloc[0]['horizon_days']
        vals.append(f"{m} (h={h_d}d): {p_val:.2f}")
    print(f"{rid:12s} | " + " | ".join(vals))
