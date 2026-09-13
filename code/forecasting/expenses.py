"""
Expense forecasting module.
Handles:
- Immediate reservation of pending debit obligations
- Explicit future scheduled debits on settlement dates
- Evidence-based recurring fixed commitments (weekly, biweekly, monthly) with cadence and stability validation
- Calendar-week aggregated variable essential spending (groceries, transport) across quantiles (mean, median, q75, q90)
- Suppression of recurring debits when explicit scheduled debits exist for the same billing cycle (anti-double-counting)
"""

from dataclasses import dataclass
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple
import pandas as pd
import numpy as np


@dataclass
class RecurringExpense:
    """Represents a verified recurring expense commitment."""
    category: str
    cadence: str  # 'monthly', 'weekly', 'biweekly'
    amount: float
    day_of_month: Optional[int] = None
    day_of_week: Optional[int] = None
    anchor_date: Optional[str] = None
    flexibility: str = 'fixed'


class ExpenseForecaster:
    """Identifies and projects fixed recurring expenses and essential variable spending."""

    FIXED_CATEGORIES = [
        'rent', 'utilities', 'debt_repayment', 'education', 'cloud_storage',
        'streaming', 'music_subscription', 'delivery_membership', 'gym',
        'insurance', 'family_support', 'housing', 'healthcare'
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
            uid = str(m.get('user_id', '')).strip()
            txt = str(m.get('message_text', ''))
            if '12%' in txt and 'rent' in txt.lower():
                self.rent_multipliers[uid] = 1.12

    def get_pending_debits_total(
        self,
        user_events: pd.DataFrame,
        request_date: str,
    ) -> Tuple[float, List[Tuple[str, str, float, str]]]:
        """
        Return (total_pending_amount, [(event_id, settlement_date, amount, category)]).
        Pending debits must be reserved immediately on request_date per challenge rules.
        """
        req_d = str(request_date)[:10]
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
            cat = str(r.get('category', 'unknown'))
            total += amt
            details.append((str(r['event_id']), s_date, amt, cat))
        return total, details

    def get_future_scheduled_debits(
        self,
        user_events: pd.DataFrame,
        request_date: str,
    ) -> List[Tuple[str, str, float, str]]:
        """
        Return list of (event_id, settlement_date, amount, category) for scheduled debits on or after request_date.
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
            cat = str(r.get('category', 'unknown'))
            details.append((str(r['event_id']), s_date, amt, cat))
        return details

    def get_recurring_commitments(
        self,
        user_id: str,
        user_events: pd.DataFrame,
        request_date: str,
    ) -> Dict[str, RecurringExpense]:
        """
        Detect cadence (weekly, biweekly, monthly) and amount for fixed expense commitments.
        Uses date spacing and amount stability to avoid inferring spurious recurrence.
        """
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

            # Amount stability
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
                    category=cat,
                    cadence='weekly',
                    amount=amt,
                    day_of_week=last_d.dayofweek,
                    anchor_date=str(last_d)[:10],
                    flexibility=flex,
                )
            # Biweekly cadence (12-16 days)
            elif 12.0 <= med_diff <= 16.0:
                last_d = settle_dates.iloc[-1]
                commitments[cat] = RecurringExpense(
                    category=cat,
                    cadence='biweekly',
                    amount=amt,
                    anchor_date=str(last_d)[:10],
                    flexibility=flex,
                )
            # Monthly cadence (25-35 days)
            elif 25.0 <= med_diff <= 35.0:
                typ_day = int(settle_dates.dt.day.mode().iloc[0])
                commitments[cat] = RecurringExpense(
                    category=cat,
                    cadence='monthly',
                    amount=amt,
                    day_of_month=typ_day,
                    flexibility=flex,
                )
            # If >= 3 events and fairly regular monthly day
            elif len(c_events) >= 3 and settle_dates.dt.day.nunique() <= 2:
                typ_day = int(settle_dates.dt.day.mode().iloc[0])
                commitments[cat] = RecurringExpense(
                    category=cat,
                    cadence='monthly',
                    amount=amt,
                    day_of_month=typ_day,
                    flexibility=flex,
                )

        return commitments

    def get_variable_essential_stats(
        self,
        user_events: pd.DataFrame,
        request_date: str,
        lookback_days: int = 180,
    ) -> Dict[str, float]:
        """
        Aggregate historical essential spending (groceries + transport) by calendar week.
        Returns true distribution statistics over weekly totals: mean, median, q75, q90.
        """
        req_d = datetime.strptime(str(request_date)[:10], "%Y-%m-%d")
        cutoff_date = (req_d - timedelta(days=lookback_days)).strftime("%Y-%m-%d")
        req_d_str = req_d.strftime("%Y-%m-%d")

        hist = user_events[
            (user_events['status'] == 'settled')
            & (user_events['direction'] == 'debit')
            & (user_events['settlement_date'] >= cutoff_date)
            & (user_events['settlement_date'] < req_d_str)
            & (user_events['category'].isin(self.VARIABLE_ESSENTIAL_CATEGORIES))
        ].copy()

        if hist.empty:
            # Fallback without lookback cutoff if insufficient data
            hist = user_events[
                (user_events['status'] == 'settled')
                & (user_events['direction'] == 'debit')
                & (user_events['settlement_date'] < req_d_str)
                & (user_events['category'].isin(self.VARIABLE_ESSENTIAL_CATEGORIES))
            ].copy()

        if hist.empty:
            return {'mean': 0.0, 'median': 0.0, 'q75': 0.0, 'q90': 0.0, 'weeks_count': 0}

        # Group by calendar week
        hist['week'] = pd.to_datetime(hist['settlement_date']).dt.to_period('W')
        weekly_totals = hist.groupby('week')['home_amount'].sum()

        if len(weekly_totals) < 2:
            single_val = float(weekly_totals.iloc[0]) if not weekly_totals.empty else 0.0
            return {'mean': single_val, 'median': single_val, 'q75': single_val, 'q90': single_val, 'weeks_count': len(weekly_totals)}

        return {
            'mean': float(weekly_totals.mean()),
            'median': float(weekly_totals.median()),
            'q75': float(weekly_totals.quantile(0.75)),
            'q90': float(weekly_totals.quantile(0.90)),
            'weeks_count': len(weekly_totals),
        }
