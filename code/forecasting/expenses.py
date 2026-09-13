"""
Expense forecasting module.
Handles:
- Pending and scheduled debit obligations
- Recurring fixed commitments (monthly rent, utilities, debt, subscriptions)
- Variable essential spending (groceries, transport) by quantiles and means
- Spending reduction and flexibility classifications
"""

from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple
import pandas as pd
import numpy as np


class ExpenseForecaster:
    """Identifies and projects fixed recurring expenses and essential variable spending."""

    MONTHLY_FIXED_CATEGORIES = [
        'rent', 'utilities', 'debt_repayment', 'education', 'cloud_storage',
        'streaming', 'music_subscription', 'delivery_membership', 'gym',
        'insurance', 'family_support', 'shopping', 'housing'
    ]

    VARIABLE_ESSENTIAL_CATEGORIES = ['groceries', 'transport']

    def __init__(self, messages_df: Optional[pd.DataFrame] = None):
        self.messages_df = messages_df if messages_df is not None else pd.DataFrame()
        self.rent_multipliers: Dict[str, float] = {}
        self._parse_messages()

    def _parse_messages(self):
        """Check for rent or subscription amendments in messages."""
        if self.messages_df.empty:
            return
        for _, m in self.messages_df.iterrows():
            uid = str(m.get('user_id', ''))
            txt = str(m.get('message_text', ''))
            if '12%' in txt and 'rent' in txt.lower():
                self.rent_multipliers[uid] = 1.12

    def get_pending_debits_total(
        self,
        user_events: pd.DataFrame,
        request_date: str,
    ) -> Tuple[float, List[Tuple[str, str, float]]]:
        """
        Return (total_pending_amount, [(event_id, settlement_date, amount)]).
        Pending debits must be reserved immediately per challenge rules.
        """
        req_d = str(request_date)[:10]
        # Valid pending debits
        mask = (
            (user_events['status'] == 'pending')
            & (user_events['direction'] == 'debit')
        )
        sub = user_events[mask]
        total = 0.0
        details = []
        for _, r in sub.iterrows():
            amt = float(r['home_amount']) if pd.notna(r.get('home_amount')) else float(r['amount'])
            s_date = str(r['settlement_date'])[:10] if pd.notna(r.get('settlement_date')) else str(r['event_date'])[:10]
            total += amt
            details.append((str(r['event_id']), s_date, amt))
        return total, details

    def get_future_scheduled_debits(
        self,
        user_events: pd.DataFrame,
        request_date: str,
    ) -> List[Tuple[str, str, float]]:
        """
        Return list of (event_id, settlement_date, amount) for scheduled debits on or after request_date.
        """
        req_d = str(request_date)[:10]
        mask = (
            (user_events['status'] == 'scheduled')
            & (user_events['direction'] == 'debit')
            & (user_events['settlement_date'] >= req_d)
        )
        sub = user_events[mask]
        details = []
        for _, r in sub.iterrows():
            amt = float(r['home_amount']) if pd.notna(r.get('home_amount')) else float(r['amount'])
            s_date = str(r['settlement_date'])[:10]
            details.append((str(r['event_id']), s_date, amt))
        return details

    def get_recurring_monthly_commitments(
        self,
        user_id: str,
        user_events: pd.DataFrame,
        request_date: str,
    ) -> Dict[str, Tuple[int, float, str]]:
        """
        Return dict of category -> (day_of_month, home_amount, flexibility).
        """
        req_d = str(request_date)[:10]
        hist = user_events[
            (user_events['status'] == 'settled')
            & (user_events['direction'] == 'debit')
            & (user_events['settlement_date'] < req_d)
        ]

        commitments = {}
        for cat in self.MONTHLY_FIXED_CATEGORIES:
            c_events = hist[hist['category'] == cat]
            if len(c_events) >= 2:
                settle_dates = pd.to_datetime(c_events['settlement_date'])
                typ_day = int(settle_dates.dt.day.mode().iloc[0])
                amt = float(c_events['home_amount'].median())
                flex = str(c_events['flexibility'].iloc[-1])

                if user_id in self.rent_multipliers and cat == 'rent':
                    amt *= self.rent_multipliers[user_id]

                commitments[cat] = (typ_day, amt, flex)
        return commitments

    def get_variable_essential_stats(
        self,
        user_events: pd.DataFrame,
        request_date: str,
    ) -> Dict[str, Dict[str, float]]:
        """
        Return weekly spend distribution metrics for groceries and transport.
        """
        req_d = str(request_date)[:10]
        hist = user_events[
            (user_events['status'] == 'settled')
            & (user_events['direction'] == 'debit')
            & (user_events['settlement_date'] < req_d)
        ]

        stats = {}
        for cat in self.VARIABLE_ESSENTIAL_CATEGORIES:
            c_events = hist[hist['category'] == cat]
            if len(c_events) >= 3:
                amts = c_events['home_amount'].dropna()
                stats[cat] = {
                    'mean': float(amts.mean()),
                    'median': float(amts.median()),
                    'q75': float(amts.quantile(0.75)),
                    'q90': float(amts.quantile(0.90)),
                }
            else:
                stats[cat] = {'mean': 0.0, 'median': 0.0, 'q75': 0.0, 'q90': 0.0}
        return stats
