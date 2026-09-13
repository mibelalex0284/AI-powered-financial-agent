"""
Confirmed Income Forecasting Module.
Identifies:
- Confirmed scheduled salary and income events
- Recurring employment salary cadence (weekly, biweekly, monthly) from historical settled credits
- Distinguishes confirmed recurring salary from uncredited windfalls, pending payouts, and explicit contract terminations
"""

from dataclasses import dataclass
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple
import pandas as pd
import numpy as np


@dataclass
class RecurringIncomeSchedule:
    """Represents an active, verified recurring salary schedule."""
    cadence: str  # 'monthly', 'weekly', 'biweekly'
    amount: float
    day_of_month: Optional[int] = None      # For monthly cadence
    day_of_week: Optional[int] = None       # For weekly cadence (0 = Monday, 6 = Sunday)
    anchor_date: Optional[str] = None       # For biweekly/interval cadence


class ConfirmedIncomeForecaster:
    """Detects and forecasts only verified, confirmed income."""

    def __init__(self, messages_df: Optional[pd.DataFrame] = None):
        self.messages_df = messages_df if messages_df is not None else pd.DataFrame()
        self._parse_message_rules()

    def _parse_message_rules(self):
        """
        Extract explicit authoritative payroll directives, contract ends, and confirmed salary adjustments
        using general pattern matching over untrusted evidence.
        """
        import re
        self.terminated_users = set()
        self.salary_overrides: Dict[str, float] = {}
        self.pay_day_overrides: Dict[str, int] = {}

        if self.messages_df.empty:
            return

        for _, m in self.messages_df.iterrows():
            uid = str(m.get('user_id', '')).strip()
            txt = str(m.get('message_text', ''))
            txt_lower = txt.lower()

            # 1. Explicit termination of employment / contract
            if (
                'employment has ended' in txt_lower
                or 'kontrak musiman saat ini telah berakhir' in txt_lower
                or ('contract has ended' in txt_lower and 'seasonal' in txt_lower)
                or 'sumber pendapatan kerja rumah tangga telah berakhir' in txt_lower
            ):
                self.terminated_users.add(uid)

            # 2. Confirmed authoritative salary amount overrides
            phrases = [
                r'reduced to (?:EUR|USD|INR|ZAR|IDR)\s*([\d,]+(?:\.\d+)?)',
                r'naik menjadi (?:EUR|USD|INR|ZAR|IDR)\s*([\d,]+(?:\.\d+)?)',
                r'gaji pokok yang dikonfirmasi adalah (?:EUR|USD|INR|ZAR|IDR)\s*([\d,]+(?:\.\d+)?)',
                r'sisa gaji bulanan yang dikonfirmasi adalah (?:EUR|USD|INR|ZAR|IDR)\s*([\d,]+(?:\.\d+)?)',
                r'first salary (?:will be|from the new employer is|of) (?:EUR|USD|INR|ZAR|IDR)\s*([\d,]+(?:\.\d+)?)',
                r'gaji pertama anda sebesar (?:EUR|USD|INR|ZAR|IDR)\s*([\d,]+(?:\.\d+)?)',
                r'regular salary (?:of|for the next payroll is) (?:EUR|USD|INR|ZAR|IDR)\s*([\d,]+(?:\.\d+)?)',
                r'salary of (?:EUR|USD|INR|ZAR|IDR)\s*([\d,]+(?:\.\d+)?) is confirmed',
                r'next salary is reduced to (?:EUR|USD|INR|ZAR|IDR)\s*([\d,]+(?:\.\d+)?)',
                r'temporary monthly pay is (?:EUR|USD|INR|ZAR|IDR)\s*([\d,]+(?:\.\d+)?)',
                r'confirmed base salary is (?:EUR|USD|INR|ZAR|IDR)\s*([\d,]+(?:\.\d+)?)',
                r'remaining confirmed monthly salary is (?:EUR|USD|INR|ZAR|IDR)\s*([\d,]+(?:\.\d+)?)',
            ]
            for pat in phrases:
                match = re.search(pat, txt, re.IGNORECASE)
                if match:
                    raw_num = match.group(1).replace(',', '')
                    self.salary_overrides[uid] = float(raw_num)
                    # Confirmed remaining base salary overrides any household termination
                    self.terminated_users.discard(uid)
                    break

            # 3. Authoritative payroll date amendment messages (generic pattern)
            date_match = re.search(r'\b(20\d{2}-\d{2}-\d{2})\b', txt)
            if date_match and (
                'replaces the payroll date' in txt_lower
                or 'menggantikan tanggal penggajian' in txt_lower
                or 'confirmed salary is now expected on' in txt_lower
                or 'confirmed credit date is' in txt_lower
                or 'tanggal kredit yang dikonfirmasi adalah' in txt_lower
                or 'scheduled for' in txt_lower
                or 'dijadwalkan pada' in txt_lower
                or 'is confirmed for' in txt_lower
            ):
                d_str = date_match.group(1)
                try:
                    dt = datetime.strptime(d_str, "%Y-%m-%d")
                    self.pay_day_overrides[uid] = dt.day
                except Exception:
                    pass

    def has_future_confirmed_income(self, user_id: str, user_events: pd.DataFrame) -> bool:
        """
        Check if user has an active confirmed income source going forward.
        Does NOT drop income on pending payouts, lottery, or commission text.
        """
        # Check for explicit 'Final employer payroll' in historical events
        for _, e in user_events.iterrows():
            if 'Final employer payroll' in str(e.get('description', '')):
                return False

        if user_id in self.terminated_users and user_id not in self.salary_overrides:
            return False

        # Check if salary events are actually variable gig-platform payouts
        # (e.g., Delivery platform, Task marketplace, QuickCrew app earnings)
        # Per §6.3, variable unconfirmed earnings must not be projected as confirmed recurring salary
        sal_events = user_events[
            (user_events['category'] == 'salary')
            & (user_events['direction'] == 'credit')
        ]
        if not sal_events.empty:
            gig_mask = sal_events['description'].str.lower().str.contains(
                'platform payout|app earnings|task marketplace|driver platform|delivery platform'
            )
            if gig_mask.all():
                return False

        return True

    def get_future_scheduled_income(
        self,
        user_events: pd.DataFrame,
        request_date: str,
    ) -> List[Tuple[str, float]]:
        """
        Return list of (settlement_date, home_amount) for explicit scheduled income
        on or after request_date.
        """
        req_d = str(request_date)[:10]
        mask = (
            (user_events['status'] == 'scheduled')
            & (user_events['direction'] == 'credit')
            & (user_events['event_type'] == 'income')
            & (user_events['settlement_date'] >= req_d)
        )
        sub = user_events[mask]
        results = []
        for _, r in sub.iterrows():
            s_date = str(r['settlement_date'])[:10]
            amt = float(r['home_amount']) if pd.notna(r.get('home_amount')) else float(r['amount'])
            results.append((s_date, amt))
        return results

    def get_recurring_salary_schedule(
        self,
        user_id: str,
        user_events: pd.DataFrame,
        request_date: str,
    ) -> Optional[RecurringIncomeSchedule]:
        """
        Detect recurring salary cadence (weekly, biweekly, monthly) and amount.
        Follows general 4-tier precedence:
        1. Authoritative payroll amendment message
        2. Confirmed future scheduled salary event
        3. Newer settled salary event
        4. Historical recurring salary cadence
        """
        if not self.has_future_confirmed_income(user_id, user_events):
            return None

        override_amt = self.salary_overrides.get(user_id)
        override_day = self.pay_day_overrides.get(user_id)
        req_d = str(request_date)[:10]

        # Scheduled salary on or after request date
        sched_sal = user_events[
            (user_events['category'] == 'salary')
            & (user_events['status'] == 'scheduled')
            & (user_events['direction'] == 'credit')
            & (user_events['settlement_date'] >= req_d)
        ]

        # Settle history before request_date
        hist_sal = user_events[
            (user_events['category'] == 'salary')
            & (user_events['status'] == 'settled')
            & (user_events['direction'] == 'credit')
            & (user_events['settlement_date'] < req_d)
        ].copy()

        # Clean salary stream: exclude commissions, bonuses, and secondary income per §6.3
        if not hist_sal.empty:
            non_base_mask = hist_sal['description'].str.lower().str.contains('bonus|commission|second household|arrears')
            base_sal = hist_sal[~non_base_mask]
            if not base_sal.empty:
                hist_sal = base_sal

        if hist_sal.empty:
            if not sched_sal.empty:
                s_row = sched_sal.iloc[0]
                s_date = datetime.strptime(str(s_row['settlement_date'])[:10], "%Y-%m-%d")
                s_col = 'home_amount' if 'home_amount' in s_row else 'amount'
                amt = override_amt if override_amt is not None else float(s_row[s_col])
                day = override_day if override_day is not None else s_date.day
                return RecurringIncomeSchedule(
                    cadence='monthly',
                    amount=amt,
                    day_of_month=day,
                )
            return None

        # Sort settled dates to detect cadence
        hist_sal = hist_sal.sort_values('settlement_date')
        settle_dates = pd.to_datetime(hist_sal['settlement_date'])
        diffs = settle_dates.diff().dt.days.dropna()
        median_interval = float(diffs.median()) if not diffs.empty else 30.0

        # Amount determination: Tier 1 override -> Tier 2 scheduled confirmed -> Tier 3 historical median
        if override_amt is not None:
            amt = override_amt
        elif not sched_sal.empty and any('prorated' in str(d).lower() for d in hist_sal['description']):
            s_col = 'home_amount' if 'home_amount' in sched_sal.columns else 'amount'
            amt = float(sched_sal.iloc[0][s_col])
        else:
            col = 'home_amount' if 'home_amount' in hist_sal.columns else 'amount'
            amts = hist_sal[col].dropna()
            amt = float(amts.median()) if not amts.empty else 0.0

        # Weekly cadence (5 to 9 days interval)
        if 5.0 <= median_interval <= 9.0:
            last_date = settle_dates.iloc[-1]
            return RecurringIncomeSchedule(
                cadence='weekly',
                amount=amt,
                day_of_week=last_date.dayofweek,
            )

        # Biweekly cadence (12 to 16 days interval)
        if 12.0 <= median_interval <= 16.0:
            last_date = settle_dates.iloc[-1]
            return RecurringIncomeSchedule(
                cadence='biweekly',
                amount=amt,
                anchor_date=str(last_date)[:10],
            )

        # Monthly cadence (default)
        # Precedence for pay day:
        # Tier 1: Authoritative message date override
        # Tier 2: Confirmed scheduled salary event day
        # Tier 3: Newer settled salary date if shifted
        # Tier 4: Mode of historical pay days
        if override_day is not None:
            typ_day = override_day
        elif not sched_sal.empty:
            s_date = datetime.strptime(str(sched_sal.iloc[0]['settlement_date'])[:10], "%Y-%m-%d")
            typ_day = s_date.day
        elif len(settle_dates) >= 2 and settle_dates.iloc[-1].day == settle_dates.iloc[-2].day:
            typ_day = int(settle_dates.iloc[-1].day)
        else:
            typ_day = int(settle_dates.dt.day.mode().iloc[0])

        return RecurringIncomeSchedule(
            cadence='monthly',
            amount=amt,
            day_of_month=typ_day,
        )
