"""
Decision Engine for HackerRank Orchestrate 'Buy or Wait?'.
Executes the full evaluation pipeline for each financial request:
1. Reconstructs user financial position and cash flows
2. Computes conservative safe-to-pay headroom
3. Determines earliest safe full-payment date
4. Evaluates all eligible candidate plans (immediate, installments, partial, wait)
5. Applies compliant spending change optimizations
6. Ranks safe plans by the exact challenge specification
7. Produces grounded, audit-compliant decision explanations
"""

from dataclasses import dataclass
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Set, Tuple
import pandas as pd
import numpy as np

from code.forecasting import (
    EventNormalizer,
    ConfirmedIncomeForecaster,
    ExpenseForecaster,
    DailyCashFlowSimulator,
)
from code.optimization.spending_changes import SpendingChangeOptimizer, SpendingChangePlan
from code.optimization.payment_optimizer import Payment, CandidatePlan, PlanSafetyEvaluator, UserBaselineCashflows


@dataclass
class DecisionResult:
    """The required output fields for a single request row."""
    request_id: str
    amount_safe_to_pay: float
    affordability_status: str
    recommended_payment_method: str
    payment_plan: str
    earliest_date_for_full_payment: str
    spending_changes_needed: str
    decision_explanation: str


class DecisionEngine:
    """Orchestrates end-to-end financial decisioning per request."""

    def __init__(
        self,
        event_normalizer: EventNormalizer,
        income_forecaster: ConfirmedIncomeForecaster,
        expense_forecaster: ExpenseForecaster,
        variable_spending_stat: str = 'median',
    ):
        self.normalizer = event_normalizer
        self.income_forecaster = income_forecaster
        self.expense_forecaster = expense_forecaster
        self.stat_key = variable_spending_stat

        self.evaluator = PlanSafetyEvaluator(income_forecaster, expense_forecaster, variable_spending_stat)
        self.spending_optimizer = SpendingChangeOptimizer()
        self.simulator = DailyCashFlowSimulator(income_forecaster, expense_forecaster, variable_spending_stat, forecast_horizon_days=90)

    def evaluate_request(
        self,
        request_row: pd.Series,
        user_profile: pd.Series,
        payment_options_df: pd.DataFrame,
    ) -> DecisionResult:
        """Process one financial request and produce the specification-compliant output row."""
        req_id = str(request_row['request_id']).strip()
        user_id = str(request_row['user_id']).strip()
        home_curr = str(user_profile.get('home_currency', 'USD')).strip()
        curr_bal = float(user_profile.get('current_available_balance', 0.0))
        min_bal = float(user_profile.get('minimum_balance_to_keep', 0.0))
        req_amt = float(request_row['requested_amount'])
        req_d_str = str(request_row['request_date'])[:10]
        comp_d_str = str(request_row['desired_completion_date'])[:10]
        allows_partial = bool(request_row.get('allows_partial_payment', False))

        user_methods = set(m.strip() for m in str(user_profile.get('payment_methods_user_will_consider', '')).split('|') if m.strip())
        max_inst_months = float(user_profile['max_installment_months']) if pd.notna(user_profile.get('max_installment_months')) else 0.0

        user_events = self.normalizer.get_user_events(user_id, home_curr, req_d_str)

        # 1. Pre-compute baseline cash flows
        baseline = self.evaluator.build_baseline_cashflows(user_id, user_events, curr_bal, min_bal, req_d_str)

        # 2. Compute amount_safe_to_pay today before optional spending changes
        sim_res = self.simulator.simulate(user_id, user_events, curr_bal, min_bal, req_amt, req_d_str)
        safe_today = sim_res.amount_safe_to_pay

        # 3. Determine earliest_date_for_full_payment
        earliest_date = self._find_earliest_full_payment_date(baseline, req_amt, req_d_str)

        # 4. Generate candidate spending change plans (0 changes first, then 1, 2, 3)
        spending_plans = self.spending_optimizer.get_candidate_plans(user_id, user_profile, user_events, req_d_str)

        candidate_plans: List[CandidatePlan] = []

        # Candidate Option A: full_payment on request_date
        if 'full_payment' in user_methods:
            for sp in spending_plans:
                # Without spending changes, full payment today requires safe_today >= req_amt
                if sp.count == 0 and safe_today < req_amt:
                    continue

                pay_full = [Payment(date=req_d_str, amount=req_amt)]
                is_s, _, _ = self.evaluator.is_plan_safe(baseline, pay_full, spending_changes=sp, horizon_days=90)
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

        # Candidate Option B: installments from request_payment_options.csv
        if 'installments' in user_methods and max_inst_months > 0:
            req_opts = payment_options_df[payment_options_df['request_id'] == req_id]
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
                    is_s, _, _ = self.evaluator.is_plan_safe(baseline, payments, spending_changes=sp, horizon_days=90)
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

        # Candidate Option C: partial_payment (exactly two payments)
        if allows_partial and ('partial_payment' in user_methods):
            if 0 < safe_today < req_amt and earliest_date is not None:
                if earliest_date <= comp_d_str:
                    p_partial = [
                        Payment(date=req_d_str, amount=safe_today),
                        Payment(date=earliest_date, amount=req_amt - safe_today)
                    ]
                    for sp in spending_plans:
                        is_s, _, _ = self.evaluator.is_plan_safe(baseline, p_partial, spending_changes=sp, horizon_days=90)
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

        # Candidate Option D: wait
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

        # 5. Plan Selection and Ranking
        # Filter plans that complete on or before desired_completion_date
        on_time_plans = [p for p in candidate_plans if p.completes_by_deadline]

        chosen_plan: Optional[CandidatePlan] = None
        if on_time_plans:
            # Ranking rules per problem statement:
            # 1. No spending changes first (count == 0)
            # 2. Minimize total amount paid
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
            # If no plan completes on time, but full payment is safe later within forecast horizon
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

        # 6. Generate grounded decision explanation
        explanation = self._generate_explanation(
            chosen_plan=chosen_plan,
            home_curr=home_curr,
            req_amt=req_amt,
            min_bal=min_bal,
            safe_today=safe_today,
            req_d_str=req_d_str,
            comp_d_str=comp_d_str
        )

        earliest_out = chosen_plan.earliest_date_for_full_payment if chosen_plan.affordability_status != 'not_affordable' else ""
        if earliest_out is None:
            earliest_out = ""

        return DecisionResult(
            request_id=req_id,
            amount_safe_to_pay=round(safe_today, 2),
            affordability_status=chosen_plan.affordability_status,
            recommended_payment_method=chosen_plan.payment_method,
            payment_plan=chosen_plan.plan_string,
            earliest_date_for_full_payment=earliest_out,
            spending_changes_needed=chosen_plan.spending_changes.spec_string,
            decision_explanation=explanation
        )

    def _find_earliest_full_payment_date(
        self,
        baseline: UserBaselineCashflows,
        req_amt: float,
        req_d_str: str,
    ) -> Optional[str]:
        """Find the earliest date where paying req_amt in full is safe."""
        req_d = datetime.strptime(req_d_str, "%Y-%m-%d")

        # Day 0 check
        pay_today = [Payment(date=req_d_str, amount=req_amt)]
        is_safe_today, _, _ = self.evaluator.is_plan_safe(baseline, pay_today, spending_changes=None, horizon_days=90)
        if is_safe_today:
            return req_d_str

        # Future days check (up to day 85 to ensure post-payment essential spend is covered)
        for d_off in range(1, 85):
            test_d = (req_d + timedelta(days=d_off)).strftime("%Y-%m-%d")
            pay_test = [Payment(date=test_d, amount=req_amt)]
            is_safe, _, _ = self.evaluator.is_plan_safe(baseline, pay_test, spending_changes=None, horizon_days=90)
            if is_safe:
                return test_d

        return None

    def _format_date(self, d_str: str) -> str:
        """Format YYYY-MM-DD into readable '15 November 2019'."""
        try:
            dt = datetime.strptime(str(d_str)[:10], "%Y-%m-%d")
            day_str = str(dt.day)
            return f"{day_str} {dt.strftime('%B %Y')}"
        except Exception:
            return d_str

    def _format_amount(self, amt: float) -> str:
        """Format number cleanly with thousands separator."""
        if abs(amt - round(amt)) < 1e-4:
            return f"{int(round(amt)):,}"
        return f"{amt:,.2f}"

    def _generate_explanation(
        self,
        chosen_plan: CandidatePlan,
        home_curr: str,
        req_amt: float,
        min_bal: float,
        safe_today: float,
        req_d_str: str,
        comp_d_str: str,
    ) -> str:
        """Produce clear, grounded explanations conforming to sample style."""
        curr = home_curr
        min_bal_fmt = self._format_amount(min_bal)
        req_amt_fmt = self._format_amount(req_amt)

        # 1. Affordable Now (Full Payment)
        if chosen_plan.affordability_status == 'affordable_now':
            return f"Pay {curr} {req_amt_fmt} today. This leaves at least {curr} {min_bal_fmt} available over the next 90 days."

        # 2. Affordable with Plan - Installments
        if chosen_plan.payment_method == 'installments':
            p_amt = self._format_amount(chosen_plan.payments[0].amount)
            start_date_fmt = self._format_date(chosen_plan.start_date)
            return (
                f"Use {chosen_plan.number_of_payments} installments of {curr} {p_amt}, "
                f"starting {start_date_fmt}. This leaves at least {curr} {min_bal_fmt} available."
            )

        # 3. Affordable with Plan - Partial Payment
        if chosen_plan.payment_method == 'partial_payment':
            amt1 = self._format_amount(chosen_plan.payments[0].amount)
            amt2 = self._format_amount(chosen_plan.payments[1].amount)
            d2_fmt = self._format_date(chosen_plan.payments[1].date)
            return (
                f"Pay {curr} {amt1} today and the remaining {curr} {amt2} on {d2_fmt}. "
                f"This completes the full request and keeps the {curr} {min_bal_fmt} minimum protected."
            )

        # 4. Affordable with Plan - Full Payment with Spending Changes
        if chosen_plan.affordability_status == 'affordable_with_plan' and chosen_plan.payment_method == 'full_payment':
            changes_desc = []
            for sc in chosen_plan.spending_changes.changes:
                if sc.action == 'stop':
                    changes_desc.append(f"stop the {sc.description.lower()}")
                elif sc.action == 'reduce_to':
                    new_amt_fmt = self._format_amount(sc.new_amount)
                    changes_desc.append(f"reduce the {sc.description.lower()} to {curr} {new_amt_fmt}")

            ch_text = " and ".join(changes_desc).capitalize()
            return f"{ch_text}, then pay {curr} {req_amt_fmt} today. This leaves at least {curr} {min_bal_fmt} available."

        # 5. Affordable Later - Wait
        if chosen_plan.payment_method == 'wait':
            wait_date_fmt = self._format_date(chosen_plan.earliest_date_for_full_payment)
            return (
                f"Pay {curr} {req_amt_fmt} in full on {wait_date_fmt}. "
                f"Paying earlier would take the balance below the {curr} {min_bal_fmt} minimum."
            )

        # 6. Not Affordable - Not Recommended
        comp_date_fmt = self._format_date(comp_d_str)
        if safe_today > 0 and (req_amt > safe_today * 2.0):
            safe_fmt = self._format_amount(safe_today)
            return (
                f"Do not proceed with the {curr} {req_amt_fmt} request. "
                f"Although {curr} {safe_fmt} is available today, the full amount cannot be completed safely within 90 days."
            )
        else:
            return (
                f"Do not make this payment by {comp_date_fmt}. "
                f"None of the available options keeps the {curr} {min_bal_fmt} minimum protected."
            )
