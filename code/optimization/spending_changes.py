"""
Spending Change Optimizer.
Identifies flexible expense reduction and stoppage actions strictly compliant with
user profile permissions, protected categories, and event-level constraints.
"""

from dataclasses import dataclass
from typing import Dict, List, Optional, Set, Tuple
import pandas as pd


@dataclass
class SpendingChange:
    """A single compliant spending adjustment."""
    action: str  # 'stop' or 'reduce_to'
    event_id: str
    category: str
    original_amount: float
    new_amount: float  # 0.0 for stop
    monthly_savings: float
    description: str

    def to_spec_string(self) -> str:
        if self.action == 'stop':
            return f"stop:{self.event_id}"
        elif self.action == 'reduce_to':
            amt_str = f"{int(self.new_amount)}" if abs(self.new_amount - round(self.new_amount)) < 1e-6 else f"{self.new_amount:.2f}"
            return f"reduce_to:{self.event_id}:{amt_str}"
        return ""


@dataclass
class SpendingChangePlan:
    """A combination of up to three non-overlapping spending changes."""
    changes: List[SpendingChange]

    @property
    def spec_string(self) -> str:
        if not self.changes:
            return "none"
        return "|".join(c.to_spec_string() for c in self.changes)

    @property
    def total_monthly_savings(self) -> float:
        return sum(c.monthly_savings for c in self.changes)

    @property
    def count(self) -> int:
        return len(self.changes)


class SpendingChangeOptimizer:
    """Finds and evaluates valid candidate spending changes for a user."""

    def __init__(self):
        pass

    def get_candidate_plans(
        self,
        user_id: str,
        user_profile: pd.Series,
        user_events: pd.DataFrame,
        request_date: str,
    ) -> List[SpendingChangePlan]:
        """
        Return candidate spending change plans ordered from 0 changes up to 3 changes.
        """
        # Parse profile categories
        protect_raw = str(user_profile.get('expense_categories_to_protect', ''))
        protected = set(c.strip() for c in protect_raw.split('|') if c.strip() and c.strip().lower() != 'nan')

        stop_raw = str(user_profile.get('expense_categories_user_is_willing_to_stop', ''))
        willing_stop = set(c.strip() for c in stop_raw.split('|') if c.strip() and c.strip().lower() != 'nan')

        reduce_raw = str(user_profile.get('expense_categories_user_is_willing_to_reduce', ''))
        willing_reduce = set(c.strip() for c in reduce_raw.split('|') if c.strip() and c.strip().lower() != 'nan')

        req_d = str(request_date)[:10]

        # Find eligible events in user_events
        # Look for events with flexibility defined
        eligible_single_actions: List[SpendingChange] = []

        # Find unique recurring/flexible events
        # Keep ONLY the latest settled event per category prior to request_date
        flex_mask = (
            user_events['flexibility'].isin(['stoppable', 'reducible', 'reducible_or_stoppable', 'flexible'])
            & (user_events['settlement_date'] < req_d)
            & (user_events['direction'] == 'debit')
        )
        flex_events = user_events[flex_mask].sort_values('settlement_date', ascending=False)

        # Map category -> latest event row
        latest_by_cat: Dict[str, pd.Series] = {}
        for _, r in flex_events.iterrows():
            cat = str(r.get('category', '')).strip()
            if cat not in latest_by_cat:
                latest_by_cat[cat] = r

        for cat, r in latest_by_cat.items():
            evt_id = str(r['event_id'])
            if cat in protected:
                continue

            amt = float(r['home_amount']) if pd.notna(r.get('home_amount')) else float(r['amount'])
            flex = str(r.get('flexibility', '')).strip()
            desc = str(r.get('description', ''))
            min_allowed = r.get('minimum_allowed_amount')

            # Option A: STOP action
            if (flex in ['stoppable', 'reducible_or_stoppable', 'flexible']) and (cat in willing_stop):
                eligible_single_actions.append(SpendingChange(
                    action='stop',
                    event_id=evt_id,
                    category=cat,
                    original_amount=amt,
                    new_amount=0.0,
                    monthly_savings=amt,
                    description=desc
                ))

            # Option B: REDUCE_TO action
            if (flex in ['reducible', 'reducible_or_stoppable', 'flexible']) and (cat in willing_reduce):
                if pd.notna(min_allowed):
                    min_amt = float(min_allowed)
                    if min_amt < amt:
                        eligible_single_actions.append(SpendingChange(
                            action='reduce_to',
                            event_id=evt_id,
                            category=cat,
                            original_amount=amt,
                            new_amount=min_amt,
                            monthly_savings=amt - min_amt,
                            description=desc
                        ))

        # Always include baseline plan with NO changes
        candidate_plans: List[SpendingChangePlan] = [SpendingChangePlan(changes=[])]

        # Add single-action plans
        for sc in eligible_single_actions:
            candidate_plans.append(SpendingChangePlan(changes=[sc]))

        # Add 2-action combinations (mutually exclusive event_id)
        n = len(eligible_single_actions)
        for i in range(n):
            for j in range(i + 1, n):
                c1 = eligible_single_actions[i]
                c2 = eligible_single_actions[j]
                if c1.event_id != c2.event_id:
                    candidate_plans.append(SpendingChangePlan(changes=[c1, c2]))

        # Add 3-action combinations (mutually exclusive event_id)
        for i in range(n):
            for j in range(i + 1, n):
                for k in range(j + 1, n):
                    c1 = eligible_single_actions[i]
                    c2 = eligible_single_actions[j]
                    c3 = eligible_single_actions[k]
                    if len({c1.event_id, c2.event_id, c3.event_id}) == 3:
                        candidate_plans.append(SpendingChangePlan(changes=[c1, c2, c3]))

        return candidate_plans
