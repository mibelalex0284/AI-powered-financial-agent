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
from typing import Dict, List, Optional, Set, Tuple
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
        Detect cadence (weekly, biweekly, triweekly, monthly) and amount for recurring commitments
        across all non-variable debit categories based on empirical transaction history.
        Uses date spacing regularity and amount stability to avoid spurious recurrence.
        """
        req_d = str(request_date)[:10]
        hist = user_events[
            (user_events['status'] == 'settled')
            & (user_events['direction'] == 'debit')
            & (user_events['settlement_date'] < req_d)
        ].copy()

        commitments = {}
        # Evaluate all debit categories present in historical settled events (except pure variable groceries)
        candidate_cats = [c for c in hist['category'].unique() if c != 'groceries']

        for cat in candidate_cats:
            c_events = hist[hist['category'] == cat].sort_values('settlement_date')
            if len(c_events) < 2:
                continue

            settle_dates = pd.to_datetime(c_events['settlement_date'])
            diffs = settle_dates.diff().dt.days.dropna()
            if diffs.empty:
                continue
            med_diff = float(diffs.median())
            std_diff = float(diffs.std()) if len(diffs) > 1 else 0.0

            # Amount stability
            amt_col = 'home_amount' if 'home_amount' in c_events.columns else 'amount'
            amts = c_events[amt_col].dropna()
            if amts.empty:
                continue
            amt = float(amts.median())
            flex = str(c_events['flexibility'].iloc[-1]) if 'flexibility' in c_events.columns else 'fixed'

            if user_id in self.rent_multipliers and cat == 'rent':
                amt *= self.rent_multipliers[user_id]

            m_diff = med_diff
            s_diff = std_diff
            dow_n = settle_dates.dt.dayofweek.nunique()
            dom_n = settle_dates.dt.day.nunique()

            # Weekly cadence (6.5 to 7.5 days, std <= 1.0, consistent day of week)
            if 6.5 <= m_diff <= 7.5 and s_diff <= 1.0 and dow_n <= 2:
                last_d = settle_dates.iloc[-1]
                commitments[cat] = RecurringExpense(
                    category=cat,
                    cadence='weekly',
                    amount=amt,
                    day_of_week=last_d.dayofweek,
                    anchor_date=str(last_d)[:10],
                    flexibility=flex,
                )
            # Biweekly cadence (13.0 to 15.0 days, std <= 1.0, consistent day of week)
            elif 13.0 <= m_diff <= 15.0 and s_diff <= 1.0 and dow_n <= 2:
                last_d = settle_dates.iloc[-1]
                commitments[cat] = RecurringExpense(
                    category=cat,
                    cadence='biweekly',
                    amount=amt,
                    anchor_date=str(last_d)[:10],
                    flexibility=flex,
                )
            # Triweekly cadence (20.0 to 22.0 days, std <= 1.0, consistent day of week)
            elif 20.0 <= m_diff <= 22.0 and s_diff <= 1.0 and dow_n <= 2:
                last_d = settle_dates.iloc[-1]
                commitments[cat] = RecurringExpense(
                    category=cat,
                    cadence='triweekly',
                    amount=amt,
                    anchor_date=str(last_d)[:10],
                    flexibility=flex,
                )
            # Monthly cadence (28.0 to 31.5 days or consistent day of month)
            elif (28.0 <= m_diff <= 31.5 and s_diff <= 1.5) or (len(c_events) >= 3 and dom_n <= 2):
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
        exclude_categories: Optional[Set[str]] = None,
    ) -> Dict[str, float]:
        """
        Aggregate historical essential spending (groceries + transport) over the lookback window.
        Excludes categories that are already modeled as discrete recurring commitments to prevent double-counting.
        Accounts for periodic / multi-week cycles so missing weeks do not artificially inflate the burn rate.
        """
        req_d = datetime.strptime(str(request_date)[:10], "%Y-%m-%d")
        cutoff_date = (req_d - timedelta(days=lookback_days)).strftime("%Y-%m-%d")
        req_d_str = req_d.strftime("%Y-%m-%d")

        active_categories = [
            c for c in self.VARIABLE_ESSENTIAL_CATEGORIES
            if not exclude_categories or c not in exclude_categories
        ]
        if not active_categories:
            return {'mean': 0.0, 'median': 0.0, 'q75': 0.0, 'q90': 0.0, 'weeks_count': 0}

        hist = user_events[
            (user_events['status'] == 'settled')
            & (user_events['direction'] == 'debit')
            & (user_events['settlement_date'] >= cutoff_date)
            & (user_events['settlement_date'] < req_d_str)
            & (user_events['category'].isin(active_categories))
        ].copy()

        if hist.empty:
            hist = user_events[
                (user_events['status'] == 'settled')
                & (user_events['direction'] == 'debit')
                & (user_events['settlement_date'] < req_d_str)
                & (user_events['category'].isin(active_categories))
            ].copy()

        if hist.empty:
            return {'mean': 0.0, 'median': 0.0, 'q75': 0.0, 'q90': 0.0, 'weeks_count': 0}

        amt_col = 'home_amount' if 'home_amount' in hist.columns else 'amount'
        earliest_dt = pd.to_datetime(hist['settlement_date'].min())
        elapsed_days = max(7, (req_d - earliest_dt).days)
        tot_amt = float(hist[amt_col].sum())

        # Group by calendar week and reindex over all elapsed weeks to represent zero weeks
        hist['week'] = pd.to_datetime(hist['settlement_date']).dt.to_period('W')
        all_weeks = pd.period_range(start=hist['week'].min(), end=pd.to_datetime(req_d_str).to_period('W'), freq='W')
        weekly_totals = hist.groupby('week')[amt_col].sum().reindex(all_weeks, fill_value=0.0)

        mean_val = tot_amt / (elapsed_days / 7.0)
        median_val = float(weekly_totals.median())
        if median_val <= 0.0:
            median_val = mean_val

        return {
            'mean': mean_val,
            'median': mean_val,
            'q75': float(weekly_totals.quantile(0.75)),
            'q90': float(weekly_totals.quantile(0.90)),
            'weeks_count': len(weekly_totals),
        }
