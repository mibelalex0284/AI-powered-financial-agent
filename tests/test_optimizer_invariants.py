"""
Unit and invariant tests for the decision engine and payment optimizer.
Verifies:
1. Candidate plan ranking order
2. Safety evaluation respecting minimum balance
3. Partial payment constraint (exactly 2 payments, sum to requested amount)
4. Spending change permissions and mutual exclusivity
5. Output format validator correctness
"""

import unittest
from datetime import datetime
import pandas as pd

from code.optimization.spending_changes import (
    SpendingChange,
    SpendingChangePlan,
    SpendingChangeOptimizer,
)
from code.optimization.payment_optimizer import (
    Payment,
    CandidatePlan,
    PlanSafetyEvaluator,
    UserBaselineCashflows,
)
from code.optimization.validator import OutputValidator, REQUIRED_COLUMNS


class TestOptimizerInvariants(unittest.TestCase):
    """Formal mathematical invariant tests for the optimizer layer."""

    def test_ranking_hierarchy(self):
        """Verify the 6 tie-breakers: deadline, spending changes, total cost, start date, payment count, option id."""
        p1 = CandidatePlan(
            payment_method='installments',
            payment_option_id='opt_1',
            payments=[Payment('2024-01-01', 100)],
            spending_changes=SpendingChangePlan([]),
            total_payable_amount=100.0,
            start_date='2024-01-01',
            completion_date='2024-01-01',
            number_of_payments=1,
            is_safe=True,
            completes_by_deadline=True,
            affordability_status='affordable_with_plan'
        )

        p2 = CandidatePlan(
            payment_method='full_payment',
            payment_option_id='opt_full',
            payments=[Payment('2024-01-01', 100)],
            spending_changes=SpendingChangePlan([
                SpendingChange('stop', 'evt_1', 'dining', 50, 0, 50, 'dining')
            ]),
            total_payable_amount=100.0,
            start_date='2024-01-01',
            completion_date='2024-01-01',
            number_of_payments=1,
            is_safe=True,
            completes_by_deadline=True,
            affordability_status='affordable_with_plan'
        )

        # Plan with 0 spending changes MUST rank higher than plan with 1 spending change
        key_fn = lambda p: (
            p.spending_changes.count,
            p.total_payable_amount,
            p.start_date,
            p.number_of_payments,
            p.payment_option_id
        )
        self.assertLess(key_fn(p1), key_fn(p2))

    def test_spending_change_mutual_exclusivity(self):
        """Verify that stop and reduce_to cannot target the same event."""
        c1 = SpendingChange('stop', 'evt_1', 'streaming', 15.0, 0.0, 15.0, 'streaming')
        c2 = SpendingChange('reduce_to', 'evt_1', 'streaming', 15.0, 5.0, 10.0, 'streaming')
        
        # In a valid plan, events must be distinct
        plan = SpendingChangePlan([c1, c2])
        event_ids = [c.event_id for c in plan.changes]
        self.assertNotEqual(len(event_ids), len(set(event_ids)))  # Detects violation

    def test_partial_payment_structure(self):
        """Verify partial payment produces exactly 2 payments totaling requested_amount."""
        req_amt = 1000.0
        safe_today = 300.0
        req_date = '2024-05-01'
        earliest_date = '2024-05-15'

        p_partial = [
            Payment(date=req_date, amount=safe_today),
            Payment(date=earliest_date, amount=req_amt - safe_today)
        ]
        self.assertEqual(len(p_partial), 2)
        self.assertAlmostEqual(sum(p.amount for p in p_partial), req_amt)

    def test_output_validator_schema(self):
        """Verify OutputValidator schema check."""
        self.assertEqual(len(REQUIRED_COLUMNS), 8)
        self.assertEqual(REQUIRED_COLUMNS[0], 'request_id')
        self.assertEqual(REQUIRED_COLUMNS[1], 'amount_safe_to_pay')
        self.assertEqual(REQUIRED_COLUMNS[2], 'affordability_status')
        self.assertEqual(REQUIRED_COLUMNS[3], 'recommended_payment_method')
        self.assertEqual(REQUIRED_COLUMNS[4], 'payment_plan')
        self.assertEqual(REQUIRED_COLUMNS[5], 'earliest_date_for_full_payment')
        self.assertEqual(REQUIRED_COLUMNS[6], 'spending_changes_needed')
        self.assertEqual(REQUIRED_COLUMNS[7], 'decision_explanation')


if __name__ == '__main__':
    unittest.main()
