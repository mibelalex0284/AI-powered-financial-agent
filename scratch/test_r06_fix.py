import sys
sys.path.insert(0, '.')
import pandas as pd
from scratch.trace_forensic_targets import normalizer, engine, profile_map, options_df
from code.optimization.payment_optimizer import Payment

gt = pd.read_csv('dataset/sample_requests.csv')
r06 = gt[gt['request_id'] == 'request_06'].iloc[0]
uid = 'user_06'
prof = profile_map[uid]
req_d = '2026-01-03'
req_amt = 620.4
ue = normalizer.get_user_events(uid, 'EUR', req_d)

old_get_rec = engine.expense_forecaster.get_recurring_commitments
def new_get_rec(u_id, u_events, rd):
    recs = old_get_rec(u_id, u_events, rd)
    # Include FIXED_CATEGORIES or transport or categories that are flexible
    return {k: v for k, v in recs.items() if (k in engine.expense_forecaster.FIXED_CATEGORIES or k == 'transport' or v.flexibility != 'fixed')}

engine.expense_forecaster.get_recurring_commitments = new_get_rec
engine.evaluator.expense_forecaster.get_recurring_commitments = new_get_rec

base = engine.evaluator.build_baseline_cashflows(uid, ue, float(prof['current_available_balance']), float(prof['minimum_balance_to_keep']), req_d)
print("Recurring commitments in base:")
for rc in base.recurring_commitments:
    print(" ", rc)

sp_plans = engine.spending_optimizer.get_candidate_plans(uid, prof, ue, req_d)
for sp in sp_plans:
    safe, min_b, min_b_d = engine.evaluator.is_plan_safe(base, [Payment(req_d, req_amt)], spending_changes=sp)
    print(f"SP: {sp.spec_string} -> safe: {safe}, min_b={min_b:.2f} on {min_b_d}")

res = engine.evaluate_request(r06, prof, options_df)
print("\nDECISION FOR REQUEST_06:")
print(f"Status: {res.affordability_status} (GT: {r06['affordability_status']})")
print(f"Method: {res.recommended_payment_method} (GT: {r06['recommended_payment_method']})")
print(f"Plan:   {res.payment_plan} (GT: {r06['payment_plan']})")
print(f"Spend:  {res.spending_changes_needed} (GT: {r06['spending_changes_needed']})")
print(f"Date:   {res.earliest_date_for_full_payment} (GT: {r06['earliest_date_for_full_payment']})")
