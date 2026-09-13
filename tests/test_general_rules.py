"""
Unit tests for general forecasting, salary precedence, and FX rules.
Verifies specification compliance on synthetic/general data without hardcoded request IDs or benchmarks.
"""

import unittest
import pandas as pd
import numpy as np
from datetime import datetime, timedelta

from code.forecasting.expenses import ExpenseForecaster, RecurringExpense
from code.forecasting.income import ConfirmedIncomeForecaster, RecurringIncomeSchedule
from code.forecasting.normalization import ExchangeRateProvider, EventNormalizer, RateResolution
from code.optimization.spending_changes import SpendingChangeOptimizer


class TestGeneralRecurrenceRules(unittest.TestCase):
    """Tests that recurring expense detection is evidence-based and not category-whitelisted."""

    def setUp(self):
        self.forecaster = ExpenseForecaster()

    def _generate_synthetic_series(self, category: str, dates: list, amounts: list, flex: str = 'fixed') -> pd.DataFrame:
        rows = []
        for i, (d, a) in enumerate(zip(dates, amounts)):
            rows.append({
                'event_id': f'evt_{category}_{i}',
                'user_id': 'user_test',
                'event_type': 'expense',
                'description': f'{category} expense',
                'category': category,
                'direction': 'debit',
                'amount': a,
                'home_amount': a,
                'settlement_date': d,
                'event_date': d,
                'status': 'settled',
                'flexibility': flex,
            })
        return pd.DataFrame(rows)

    def test_recurring_dining_can_enter_baseline(self):
        """Discretionary dining with regular biweekly intervals enters baseline."""
        dates = ['2025-01-01', '2025-01-15', '2025-01-29', '2025-02-12']
        amounts = [60.0, 62.0, 61.0, 60.5]
        events = self._generate_synthetic_series('dining', dates, amounts, flex='reducible')

        recs = self.forecaster.get_recurring_commitments('user_test', events, '2025-03-01')
        self.assertIn('dining', recs)
        self.assertEqual(recs['dining'].cadence, 'biweekly')
        self.assertEqual(recs['dining'].flexibility, 'reducible')
        self.assertAlmostEqual(recs['dining'].amount, 60.75, places=2)

    def test_recurring_shopping_can_enter_baseline(self):
        """Shopping with regular monthly cadence enters baseline."""
        dates = ['2024-10-12', '2024-11-12', '2024-12-12', '2025-01-12']
        amounts = [40.0, 40.0, 40.0, 40.0]
        events = self._generate_synthetic_series('shopping', dates, amounts, flex='stoppable')

        recs = self.forecaster.get_recurring_commitments('user_test', events, '2025-02-01')
        self.assertIn('shopping', recs)
        self.assertEqual(recs['shopping'].cadence, 'monthly')
        self.assertEqual(recs['shopping'].day_of_month, 12)
        self.assertEqual(recs['shopping'].flexibility, 'stoppable')

    def test_recurring_entertainment_can_enter_baseline(self):
        """Entertainment on a consistent monthly date enters baseline."""
        dates = ['2024-11-14', '2024-12-14', '2025-01-14', '2025-02-14']
        amounts = [30.0, 31.0, 30.5, 31.5]
        events = self._generate_synthetic_series('entertainment', dates, amounts)

        recs = self.forecaster.get_recurring_commitments('user_test', events, '2025-03-01')
        self.assertIn('entertainment', recs)
        self.assertEqual(recs['entertainment'].cadence, 'monthly')
        self.assertEqual(recs['entertainment'].day_of_month, 14)

    def test_recurring_transport_can_enter_baseline(self):
        """Transport with weekly transit cadence enters baseline."""
        dates = ['2025-01-07', '2025-01-14', '2025-01-21', '2025-01-28']
        amounts = [25.0, 25.0, 25.0, 25.0]
        events = self._generate_synthetic_series('transport', dates, amounts)

        recs = self.forecaster.get_recurring_commitments('user_test', events, '2025-02-01')
        self.assertIn('transport', recs)
        self.assertEqual(recs['transport'].cadence, 'weekly')

    def test_irregular_discretionary_spending_does_not_become_recurring(self):
        """Random erratic discretionary expenses must NOT be detected as recurring."""
        dates = ['2025-01-02', '2025-01-05', '2025-01-22', '2025-02-18']
        amounts = [15.0, 120.0, 45.0, 8.0]
        events = self._generate_synthetic_series('shopping', dates, amounts)

        recs = self.forecaster.get_recurring_commitments('user_test', events, '2025-03-01')
        self.assertNotIn('shopping', recs)

    def test_flexible_recurring_remains_eligible_for_spending_optimization(self):
        """Flexible recurring commitments in baseline are eligible for optimization in spending optimizer."""
        dates = ['2025-01-01', '2025-01-15', '2025-01-29']
        amounts = [100.0, 100.0, 100.0]
        events = self._generate_synthetic_series('dining', dates, amounts, flex='reducible')
        events['minimum_allowed_amount'] = 50.0

        profile = pd.Series({
            'user_id': 'user_test',
            'expense_categories_to_protect': 'rent|groceries',
            'expense_categories_user_is_willing_to_reduce': 'dining',
            'expense_categories_user_is_willing_to_stop': '',
        })

        sp_opt = SpendingChangeOptimizer()
        plans = sp_opt.get_candidate_plans('user_test', profile, events, '2025-02-01')

        # Baseline plan plus reduction plan
        spec_strings = [p.spec_string for p in plans]
        self.assertIn('none', spec_strings)
        self.assertTrue(any('reduce_to:evt_dining_2:50' in s for s in spec_strings))


class TestGeneralSalaryPrecedenceRules(unittest.TestCase):
    """Tests the 4-tier salary precedence hierarchy."""

    def setUp(self):
        pass

    def test_authoritative_payroll_amendment_overrides_historical_cadence(self):
        """Tier 1: Explicit payroll amendment date message overrides historical date."""
        messages = pd.DataFrame([{
            'message_id': 'msg_01',
            'user_id': 'u_sal',
            'sent_at': '2025-01-20T00:00:00Z',
            'source_type': 'employer',
            'message_text': 'Your confirmed salary is now expected on 2025-02-23. This replaces the payroll date shown earlier.'
        }])
        inc_fc = ConfirmedIncomeForecaster(messages)
        self.assertEqual(inc_fc.pay_day_overrides.get('u_sal'), 23)

        # Historical events had salary on 15th
        hist_events = pd.DataFrame([
            {'user_id': 'u_sal', 'category': 'salary', 'status': 'settled', 'direction': 'credit', 'amount': 2000.0, 'home_amount': 2000.0, 'settlement_date': '2024-11-15', 'description': 'Salary'},
            {'user_id': 'u_sal', 'category': 'salary', 'status': 'settled', 'direction': 'credit', 'amount': 2000.0, 'home_amount': 2000.0, 'settlement_date': '2024-12-15', 'description': 'Salary'},
            {'user_id': 'u_sal', 'category': 'salary', 'status': 'settled', 'direction': 'credit', 'amount': 2000.0, 'home_amount': 2000.0, 'settlement_date': '2025-01-15', 'description': 'Salary'},
        ])
        sched = inc_fc.get_recurring_salary_schedule('u_sal', hist_events, '2025-02-01')
        self.assertIsNotNone(sched)
        self.assertEqual(sched.day_of_month, 23)  # Overridden to 23

    def test_future_confirmed_scheduled_salary_beats_historical_cadence(self):
        """Tier 2: Future scheduled salary event day takes precedence over historical mode."""
        inc_fc = ConfirmedIncomeForecaster(pd.DataFrame())
        events = pd.DataFrame([
            {'user_id': 'u_sal2', 'category': 'salary', 'status': 'settled', 'direction': 'credit', 'amount': 3000.0, 'home_amount': 3000.0, 'settlement_date': '2024-11-10', 'description': 'Salary'},
            {'user_id': 'u_sal2', 'category': 'salary', 'status': 'settled', 'direction': 'credit', 'amount': 3000.0, 'home_amount': 3000.0, 'settlement_date': '2024-12-10', 'description': 'Salary'},
            {'user_id': 'u_sal2', 'category': 'salary', 'status': 'scheduled', 'direction': 'credit', 'amount': 3000.0, 'home_amount': 3000.0, 'settlement_date': '2025-01-25', 'description': 'Confirmed next salary'},
        ])
        sched = inc_fc.get_recurring_salary_schedule('u_sal2', events, '2025-01-01')
        self.assertIsNotNone(sched)
        self.assertEqual(sched.day_of_month, 25)

    def test_vague_unconfirmed_message_does_not_override_confirmed_income(self):
        """Unconfirmed bonus / speculative message does not create or alter confirmed recurring salary."""
        messages = pd.DataFrame([{
            'message_id': 'msg_02',
            'user_id': 'u_sal3',
            'sent_at': '2025-01-10T00:00:00Z',
            'source_type': 'employer',
            'message_text': 'Your quarterly performance bonus is still under review and has not been approved yet.'
        }])
        inc_fc = ConfirmedIncomeForecaster(messages)
        self.assertNotIn('u_sal3', inc_fc.salary_overrides)

    def test_unrelated_message_does_not_affect_salary(self):
        """Unrelated receipt or customer service message does not alter payroll date or amount."""
        messages = pd.DataFrame([{
            'message_id': 'msg_03',
            'user_id': 'u_sal4',
            'sent_at': '2025-01-10T00:00:00Z',
            'source_type': 'merchant',
            'message_text': 'Thank you for dining at Nagarjuna. Tax invoice total is INR 8,528.'
        }])
        inc_fc = ConfirmedIncomeForecaster(messages)
        self.assertNotIn('u_sal4', inc_fc.salary_overrides)
        self.assertNotIn('u_sal4', inc_fc.pay_day_overrides)


class TestGeneralFXDatedRateRules(unittest.TestCase):
    """Tests strict dated exchange rate resolution without prior-date fallback."""

    def setUp(self):
        self.rates_df = pd.DataFrame([
            {'rate_date': '2025-05-10', 'from_currency': 'USD', 'to_currency': 'EUR', 'rate': 0.92},
            {'rate_date': '2025-05-10', 'from_currency': 'EUR', 'to_currency': 'INR', 'rate': 90.0},
            {'rate_date': '2025-05-01', 'from_currency': 'USD', 'to_currency': 'GBP', 'rate': 0.78},
        ])
        self.provider = ExchangeRateProvider(self.rates_df)

    def test_exact_date_direct_rate(self):
        """Direct match on exact date succeeds."""
        res = self.provider.resolve_rate('2025-05-10', 'USD', 'EUR')
        self.assertEqual(res.match_type, 'exact_direct')
        self.assertAlmostEqual(res.rate, 0.92)

    def test_exact_date_inverse_rate(self):
        """Inverse match on exact date succeeds."""
        res = self.provider.resolve_rate('2025-05-10', 'EUR', 'USD')
        self.assertEqual(res.match_type, 'exact_inverse')
        self.assertAlmostEqual(res.rate, 1.0 / 0.92)

    def test_exact_date_triangulation(self):
        """Triangulation across exact date succeeds (USD -> EUR -> INR)."""
        res = self.provider.resolve_rate('2025-05-10', 'USD', 'INR')
        self.assertEqual(res.match_type, 'triangulated')
        self.assertAlmostEqual(res.rate, 0.92 * 90.0)

    def test_unavailable_date_does_not_silently_fallback(self):
        """Requesting an exchange rate on a date with no conversion path raises ValueError."""
        with self.assertRaises(ValueError) as ctx:
            self.provider.resolve_rate('2025-05-15', 'USD', 'EUR')
        self.assertIn('No exact dated exchange rate path found', str(ctx.exception))


if __name__ == '__main__':
    unittest.main()
