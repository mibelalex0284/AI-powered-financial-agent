import sys
sys.path.insert(0, '.')
import pandas as pd
from datetime import datetime, timedelta
from scratch.trace_forensic_targets import normalizer, engine, profile_map, options_df
from code.optimization.payment_optimizer import Payment

def trace_user_11():
    uid = 'user_11'
    prof = profile_map[uid]
    req_d = '2025-05-03'
    req_amt = 13110000.0
    min_bal = float(prof['minimum_balance_to_keep'])
    curr_bal = float(prof['current_available_balance'])
    ue = normalizer.get_user_events(uid, 'IDR', req_d)
    base = engine.evaluator.build_baseline_cashflows(uid, ue, curr_bal, min_bal, req_d)
    sp_plans = engine.spending_optimizer.get_candidate_plans(uid, prof, ue, req_d)
    sp1 = [p for p in sp_plans if 'event_989' in p.spec_string and p.count == 1][0]
    
    print("User 11 current balance:", curr_bal)
    print("User 11 min balance to keep:", min_bal)
    print("Requested:", req_amt)
    print("Daily var rate:", base.daily_var_rate)
    
    # Let's inspect variable essential stats details
    rec_cats = set(rc.category for rc in base.recurring_commitments)
    vstats = engine.expense_forecaster.get_variable_essential_stats(ue, req_d, exclude_categories=rec_cats)
    print("Var stats:", vstats)
    
    # Trace day by day with sp1
    stopped_categories = set(sc.category for sc in sp1.changes if sc.action == 'stop')
    reduced_categories = {sc.category: sc.new_amount for sc in sp1.changes if sc.action == 'reduce_to'}
    
    pay_full = [Payment(date=req_d, amount=req_amt)]
    pay_map = {p.date: p.amount for p in pay_full}
    
    cur_bal = base.initial_balance
    req_dt = datetime.strptime(req_d, "%Y-%m-%d")
    print(f"\nDay-by-day trajectory for user 11 with full payment & {sp1.spec_string}:")
    for d in range(15):
        cur_date = req_dt + timedelta(days=d)
        d_str = cur_date.strftime("%Y-%m-%d")
        income = 0.0
        expenses = 0.0
        
        # Check income
        if base.salary_schedule and cur_date.day == base.salary_schedule.day_of_month:
            income += base.salary_schedule.amount
            print(f"  {d_str}: Salary +{base.salary_schedule.amount}")
            
        # Check recurring
        for rc in base.recurring_commitments:
            amt = reduced_categories.get(rc.category, rc.amount)
            if rc.category in stopped_categories:
                continue
            is_due = False
            if rc.cadence == 'monthly' and cur_date.day == rc.day_of_month:
                is_due = True
            elif rc.cadence == 'weekly' and cur_date.weekday() == rc.day_of_week:
                is_due = True
            elif rc.cadence == 'biweekly' and rc.anchor_date:
                anchor = datetime.strptime(rc.anchor_date, "%Y-%m-%d")
                if (cur_date - anchor).days > 0 and (cur_date - anchor).days % 14 == 0:
                    is_due = True
            elif rc.cadence == 'triweekly' and rc.anchor_date:
                anchor = datetime.strptime(rc.anchor_date, "%Y-%m-%d")
                if (cur_date - anchor).days > 0 and (cur_date - anchor).days % 21 == 0:
                    is_due = True
            if is_due:
                expenses += amt
                print(f"  {d_str}: {rc.category} ({rc.cadence}) -{amt}")
                
        # Scheduled debits
        if d_str in base.sched_debit_map:
            expenses += base.sched_debit_map[d_str]
            print(f"  {d_str}: Scheduled debit -{base.sched_debit_map[d_str]}")
            
        # Daily var
        expenses += base.daily_var_rate
        
        # Payment
        if d_str in pay_map:
            expenses += pay_map[d_str]
            print(f"  {d_str}: Payment -{pay_map[d_str]}")
            
        cur_bal = cur_bal + income - expenses
        print(f"  {d_str} (day {d}): Bal = {cur_bal:,.2f} (diff from min: {cur_bal - min_bal:,.2f})")

if __name__ == '__main__':
    trace_user_11()
