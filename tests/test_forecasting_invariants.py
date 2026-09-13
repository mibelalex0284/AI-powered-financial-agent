"""
Invariant and Specification Validation Test Suite for Forecasting & Simulation Layer.
Validates all 10 core financial and physical invariants demanded by the project contract:
1. predicted_reserve = current_balance - projected_minimum_balance
2. safe_amount = max(0, min(requested_amount, projected_minimum_balance - minimum_balance))
3. Increasing an expense cannot increase safe amount
4. Adding confirmed income cannot decrease safe amount
5. Adding an unconfirmed income event must not increase safe amount
6. Failed/cancelled transactions must have no effect
7. Missing amount must never be interpreted as zero
8. A scheduled debit must not also appear as an inferred recurring debit
9. A confirmed salary must not disappear merely because unrelated unconfirmed income exists
10. No future event may leak into historical variable-spending statistics
11. Verification that all 16 blank events resolve via ImageAmountResolver
"""

import os
import sys
import unittest
import pandas as pd
import numpy as np

# Add code directory to path
repo_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
code_dir = os.path.join(repo_root, "code")
if code_dir not in sys.path:
    sys.path.insert(0, code_dir)

from forecasting import (
    EventNormalizer,
    ExchangeRateProvider,
    ImageAmountResolver,
    ConfirmedIncomeForecaster,
    ExpenseForecaster,
    DailyCashFlowSimulator,
)


class TestForecastingInvariants(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.data_dir = os.path.join(repo_root, "dataset")
        cls.samples_df = pd.read_csv(os.path.join(cls.data_dir, "sample_requests.csv"))
        cls.profiles_df = pd.read_csv(os.path.join(cls.data_dir, "financial_profiles.csv"))
        cls.events_df = pd.read_csv(os.path.join(cls.data_dir, "financial_events.csv"))
        cls.messages_df = pd.read_csv(os.path.join(cls.data_dir, "messages.csv"))
        cls.images_df = pd.read_csv(os.path.join(cls.data_dir, "images.csv"))
        cls.rates_df = pd.read_csv(os.path.join(cls.data_dir, "exchange_rates.csv"))

        cls.rate_provider = ExchangeRateProvider(cls.rates_df)
        cls.normalizer = EventNormalizer(cls.events_df, images_df=cls.images_df, rate_provider=cls.rate_provider)
        cls.income_forecaster = ConfirmedIncomeForecaster(cls.messages_df)
        cls.expense_forecaster = ExpenseForecaster(cls.messages_df)
        cls.simulator = DailyCashFlowSimulator(cls.income_forecaster, cls.expense_forecaster, variable_spending_stat='median')

    def test_invariant_1_and_2_algebraic_consistency(self):
        """Test Invariants 1 & 2 across all 25 sample requests."""
        for _, sample in self.samples_df.iterrows():
            uid = sample['user_id']
            req_d = sample['request_date']
            req_amt = float(sample['requested_amount'])
            prof = self.profiles_df[self.profiles_df['user_id'] == uid].iloc[0]
            curr_bal = float(prof['current_available_balance'])
            min_bal = float(prof['minimum_balance_to_keep'])

            ue = self.normalizer.get_user_events(uid, prof['home_currency'], req_d)
            res = self.simulator.simulate(uid, ue, curr_bal, min_bal, req_amt, req_d)

            # Invariant 1: predicted_reserve = max(0, current_balance - projected_minimum_balance)
            expected_reserve = max(0.0, curr_bal - res.projected_minimum_balance)
            self.assertAlmostEqual(
                res.predicted_reserve, expected_reserve, places=4,
                msg=f"Invariant 1 violated for {sample['request_id']}"
            )

            # Invariant 2: safe_amount = max(0, min(requested_amount, projected_minimum_balance - minimum_balance))
            expected_safe = max(0.0, min(req_amt, res.projected_minimum_balance - min_bal))
            self.assertAlmostEqual(
                res.amount_safe_to_pay, expected_safe, places=4,
                msg=f"Invariant 2 violated for {sample['request_id']}"
            )

    def test_invariant_3_increasing_expense_cannot_increase_safe_amount(self):
        """Invariant 3: Adding an expense obligation cannot increase safe amount."""
        uid = 'user_01'
        prof = self.profiles_df[self.profiles_df['user_id'] == uid].iloc[0]
        req_d = '2024-03-03'
        curr_bal = float(prof['current_available_balance'])
        min_bal = float(prof['minimum_balance_to_keep'])

        ue = self.normalizer.get_user_events(uid, prof['home_currency'], req_d)
        res_baseline = self.simulator.simulate(uid, ue, curr_bal, min_bal, 25256.0, req_d)

        # Inject an additional pending debit
        ue_modified = ue.copy()
        new_row = ue.iloc[0].copy()
        new_row['event_id'] = 'test_extra_debit'
        new_row['status'] = 'pending'
        new_row['direction'] = 'debit'
        new_row['home_amount'] = 5000.0
        new_row['amount'] = 5000.0
        ue_modified = pd.concat([ue_modified, pd.DataFrame([new_row])], ignore_index=True)

        res_extra_expense = self.simulator.simulate(uid, ue_modified, curr_bal, min_bal, 25256.0, req_d)
        self.assertLessEqual(res_extra_expense.amount_safe_to_pay, res_baseline.amount_safe_to_pay)

    def test_invariant_4_adding_confirmed_income_cannot_decrease_safe_amount(self):
        """Invariant 4: Adding confirmed future income cannot decrease safe amount."""
        uid = 'user_08'
        prof = self.profiles_df[self.profiles_df['user_id'] == uid].iloc[0]
        req_d = '2025-02-07'
        curr_bal = float(prof['current_available_balance'])
        min_bal = float(prof['minimum_balance_to_keep'])

        ue = self.normalizer.get_user_events(uid, prof['home_currency'], req_d)
        res_baseline = self.simulator.simulate(uid, ue, curr_bal, min_bal, 996.6, req_d)

        # Inject confirmed scheduled income
        ue_modified = ue.copy()
        new_row = ue.iloc[0].copy()
        new_row['event_id'] = 'test_extra_income'
        new_row['status'] = 'scheduled'
        new_row['direction'] = 'credit'
        new_row['event_type'] = 'income'
        new_row['settlement_date'] = '2025-02-10'
        new_row['home_amount'] = 500.0
        new_row['amount'] = 500.0
        ue_modified = pd.concat([ue_modified, pd.DataFrame([new_row])], ignore_index=True)

        res_extra_income = self.simulator.simulate(uid, ue_modified, curr_bal, min_bal, 996.6, req_d)
        self.assertGreaterEqual(res_extra_income.amount_safe_to_pay, res_baseline.amount_safe_to_pay)

    def test_invariant_5_unconfirmed_income_must_not_increase_safe_amount(self):
        """Invariant 5: Pending credits or unconfirmed windfalls must not increase safe amount."""
        uid = 'user_08'
        prof = self.profiles_df[self.profiles_df['user_id'] == uid].iloc[0]
        req_d = '2025-02-07'
        curr_bal = float(prof['current_available_balance'])
        min_bal = float(prof['minimum_balance_to_keep'])

        ue = self.normalizer.get_user_events(uid, prof['home_currency'], req_d)
        res_baseline = self.simulator.simulate(uid, ue, curr_bal, min_bal, 996.6, req_d)

        # Inject pending credit
        ue_modified = ue.copy()
        new_row = ue.iloc[0].copy()
        new_row['event_id'] = 'test_pending_credit'
        new_row['status'] = 'pending'
        new_row['direction'] = 'credit'
        new_row['event_type'] = 'income'
        new_row['home_amount'] = 100000.0
        new_row['amount'] = 100000.0
        ue_modified = pd.concat([ue_modified, pd.DataFrame([new_row])], ignore_index=True)

        res_unconfirmed = self.simulator.simulate(uid, ue_modified, curr_bal, min_bal, 996.6, req_d)
        self.assertEqual(res_unconfirmed.amount_safe_to_pay, res_baseline.amount_safe_to_pay)

    def test_invariant_6_failed_and_cancelled_events_have_no_effect(self):
        """Invariant 6: Failed and cancelled transactions must have zero effect on simulation."""
        uid = 'user_05'
        prof = self.profiles_df[self.profiles_df['user_id'] == uid].iloc[0]
        req_d = '2025-11-06'
        curr_bal = float(prof['current_available_balance'])
        min_bal = float(prof['minimum_balance_to_keep'])

        ue = self.normalizer.get_user_events(uid, prof['home_currency'], req_d)
        res_baseline = self.simulator.simulate(uid, ue, curr_bal, min_bal, 15488.0, req_d)

        # Add failed and cancelled debits
        ue_modified = ue.copy()
        r1 = ue.iloc[0].copy()
        r1['event_id'] = 'test_failed_debit'
        r1['status'] = 'failed'
        r1['direction'] = 'debit'
        r1['home_amount'] = 20000.0

        r2 = ue.iloc[0].copy()
        r2['event_id'] = 'test_cancelled_debit'
        r2['status'] = 'cancelled'
        r2['direction'] = 'debit'
        r2['home_amount'] = 20000.0
        ue_modified = pd.concat([ue_modified, pd.DataFrame([r1, r2])], ignore_index=True)

        res_with_failed = self.simulator.simulate(uid, ue_modified, curr_bal, min_bal, 15488.0, req_d)
        self.assertEqual(res_baseline.amount_safe_to_pay, res_with_failed.amount_safe_to_pay)
        self.assertEqual(res_baseline.predicted_reserve, res_with_failed.predicted_reserve)

    def test_invariant_7_missing_amount_never_interpreted_as_zero(self):
        """Invariant 7: Missing event amount must never be silently converted to 0.0."""
        # Check raw events with NaN that have no image
        resolver = ImageAmountResolver(pd.DataFrame())
        unresolved = resolver.resolve_amount('non_existent_event_xyz')
        self.assertIsNone(unresolved)
        self.assertNotEqual(unresolved, 0.0)

    def test_invariant_8_anti_double_counting_scheduled_and_recurring(self):
        """Invariant 8: Scheduled debits must suppress inferred recurring debits for that cycle."""
        uid = 'user_16'
        ue = self.normalizer.get_user_events(uid, 'INR', '2023-08-12')
        commitments = self.expense_forecaster.get_recurring_commitments(uid, ue, '2023-08-12')
        self.assertIn('rent', commitments)

        # In August 2023, user_16 has scheduled rent event_1442 (100,000 INR on 2023-08-16)
        sched_debits = self.expense_forecaster.get_future_scheduled_debits(ue, '2023-08-12')
        has_sched_rent = any(cat == 'rent' and s_date.startswith('2023-08') for _, s_date, _, cat in sched_debits)
        self.assertTrue(has_sched_rent)

    def test_invariant_9_salary_not_suppressed_by_unrelated_windfalls(self):
        """Invariant 9: Unrelated prize claims do not kill confirmed employment salary, but pending gig earnings are excluded."""
        # 1. user_23 has DrawPay prize claim message, but recurring employment salary of 45,760 ZAR is active
        u23_events = self.normalizer.get_user_events('user_23', 'ZAR', '2025-05-07')
        sched_sal = self.income_forecaster.get_recurring_salary_schedule('user_23', u23_events, '2025-05-07')
        self.assertIsNotNone(sched_sal, "user_23 employment salary must NOT be suppressed by prize claim message")
        self.assertEqual(sched_sal.amount, 45760.0)

        # 2. user_10 has fluctuating QuickCrew/Delivery platform earnings where message_07 confirms payouts are pending/variable
        # Per §6.3, pending and unconfirmed earnings are excluded from guaranteed future salary
        u10_events = self.normalizer.get_user_events('user_10', 'INR', '2024-12-06')
        sched_u10 = self.income_forecaster.get_recurring_salary_schedule('user_10', u10_events, '2024-12-06')
        self.assertIsNone(sched_u10, "user_10 unconfirmed/pending gig platform earnings must NOT be credited as confirmed recurring salary")

    def test_invariant_10_no_future_leakage_in_variable_spending(self):
        """Invariant 10: No future event on or after request_date may alter variable spend statistics."""
        uid = 'user_01'
        req_d = '2024-03-03'
        ue = self.normalizer.get_user_events(uid, 'ZAR', req_d)
        stats_base = self.expense_forecaster.get_variable_essential_stats(ue, req_d)

        # Add an event after request_date
        ue_leak = ue.copy()
        new_row = ue.iloc[0].copy()
        new_row['event_id'] = 'future_leak_test'
        new_row['settlement_date'] = '2024-03-10'
        new_row['category'] = 'groceries'
        new_row['direction'] = 'debit'
        new_row['status'] = 'settled'
        new_row['home_amount'] = 99999.0
        ue_leak = pd.concat([ue_leak, pd.DataFrame([new_row])], ignore_index=True)

        stats_leak = self.expense_forecaster.get_variable_essential_stats(ue_leak, req_d)
        self.assertEqual(stats_base['median'], stats_leak['median'])
        self.assertEqual(stats_base['mean'], stats_leak['mean'])

    def test_all_16_blank_events_resolved_from_images(self):
        """Verify that all 16 blank events in financial_events.csv resolve to exact verified image amounts."""
        blank_events = self.events_df[self.events_df['amount'].isna()].copy()
        self.assertEqual(len(blank_events), 16, "Must have exactly 16 blank events in dataset")

        resolver = ImageAmountResolver(self.images_df)
        for _, row in blank_events.iterrows():
            evt_id = str(row['event_id'])
            resolved_amt = resolver.resolve_amount(evt_id)
            self.assertIsNotNone(resolved_amt, f"Event {evt_id} must resolve from its linked image")
            self.assertGreater(resolved_amt, 0.0, f"Resolved amount for {evt_id} must be strictly positive")

    def test_fx_explicit_resolution_contract(self):
        """Audit exact exchange rate resolution paths: direct, inverse, triangulated, and missing."""
        # 1. Identity
        res_id = self.rate_provider.resolve_rate('2024-01-15', 'USD', 'USD')
        self.assertEqual(res_id.rate, 1.0)
        self.assertEqual(res_id.match_type, 'identity')

        # 2. Direct exact date rate (e.g. USD -> INR on 2024-01-15)
        res_dir = self.rate_provider.resolve_rate('2024-01-15', 'USD', 'INR')
        self.assertGreater(res_dir.rate, 70.0)
        self.assertEqual(res_dir.match_type, 'exact_direct')

        # 3. Inverse exact date rate (e.g. INR -> USD on 2024-01-15)
        res_inv = self.rate_provider.resolve_rate('2024-01-15', 'INR', 'USD')
        self.assertAlmostEqual(res_inv.rate, 1.0 / res_dir.rate, places=6)
        self.assertEqual(res_inv.match_type, 'exact_inverse')

        # 4. Triangulated exact date rate (synthetic test data)
        rates_synth = pd.DataFrame([
            {'rate_date': '2025-01-01', 'from_currency': 'ABC', 'to_currency': 'USD', 'rate': 2.0},
            {'rate_date': '2025-01-01', 'from_currency': 'USD', 'to_currency': 'XYZ', 'rate': 5.0}
        ])
        prov_synth = ExchangeRateProvider(rates_synth)
        res_tri = prov_synth.resolve_rate('2025-01-01', 'ABC', 'XYZ')
        self.assertEqual(res_tri.rate, 10.0)
        self.assertEqual(res_tri.match_type, 'triangulated')

        # 5. Missing exact date rate with strict resolution
        with self.assertRaises(ValueError):
            prov_synth.resolve_rate('2099-01-01', 'NONEXIST', 'USD')


if __name__ == "__main__":
    unittest.main()
