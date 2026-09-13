"""
Validator for output.csv conforming to HackerRank Orchestrate 'Buy or Wait?'.
Validates:
1. File existence and row count (exactly 250 rows matching dataset/requests.csv)
2. Exact 8 column headers in required order
3. Valid enum values for affordability_status and recommended_payment_method
4. Mathematical invariant 0 <= amount_safe_to_pay <= requested_amount
5. Strict syntax for payment_plan, earliest_date_for_full_payment, spending_changes_needed
6. Consistency rules (e.g. affordable_now -> earliest_date == request_date)
"""

import re
from typing import Dict, List, Tuple
import pandas as pd


REQUIRED_COLUMNS = [
    'request_id',
    'amount_safe_to_pay',
    'affordability_status',
    'recommended_payment_method',
    'payment_plan',
    'earliest_date_for_full_payment',
    'spending_changes_needed',
    'decision_explanation'
]

VALID_AFFORDABILITY_STATUSES = {
    'affordable_now',
    'affordable_with_plan',
    'affordable_later',
    'not_affordable'
}

VALID_PAYMENT_METHODS = {
    'full_payment',
    'partial_payment',
    'installments',
    'wait',
    'not_recommended'
}

DATE_REGEX = re.compile(r'^\d{4}-\d{2}-\d{2}$')
PLAN_REGEX = re.compile(r'^\d{4}-\d{2}-\d{2}:\d+(\.\d+)?(\|\d{4}-\d{2}-\d{2}:\d+(\.\d+)?)*$')
SPENDING_CHANGE_REGEX = re.compile(r'^(stop:[a-zA-Z0-9_]+|reduce_to:[a-zA-Z0-9_]+:\d+(\.\d+)?)$')


class OutputValidator:
    """Validates output.csv against all challenge contract constraints."""

    @staticmethod
    def validate_file(
        output_path: str,
        requests_df: pd.DataFrame
    ) -> Tuple[bool, List[str]]:
        """
        Validate output.csv against requests_df.
        Returns (is_valid, list_of_errors).
        """
        errors: List[str] = []

        try:
            out_df = pd.read_csv(output_path, dtype=str).fillna('')
        except Exception as e:
            return False, [f"Failed to read CSV at {output_path}: {e}"]

        # 1. Column names and order
        if list(out_df.columns) != REQUIRED_COLUMNS:
            errors.append(f"Column headers mismatch. Expected: {REQUIRED_COLUMNS}, Got: {list(out_df.columns)}")

        # 2. Row count
        expected_rows = len(requests_df)
        if len(out_df) != expected_rows:
            errors.append(f"Row count mismatch. Expected {expected_rows}, Got {len(out_df)}")

        # Index requests by request_id
        req_map: Dict[str, pd.Series] = {
            str(r['request_id']).strip(): r for _, r in requests_df.iterrows()
        }

        seen_ids = set()

        for idx, row in out_df.iterrows():
            r_id = str(row.get('request_id', '')).strip()
            row_prefix = f"Row {idx+1} ({r_id}):"

            # Check uniqueness
            if not r_id:
                errors.append(f"{row_prefix} Empty request_id")
                continue
            if r_id in seen_ids:
                errors.append(f"{row_prefix} Duplicate request_id: {r_id}")
            seen_ids.add(r_id)

            if r_id not in req_map:
                errors.append(f"{row_prefix} Unknown request_id not in requests.csv")
                continue

            req_spec = req_map[r_id]
            req_amt = float(req_spec['requested_amount'])
            req_date = str(req_spec['request_date'])[:10]

            # 3. amount_safe_to_pay
            safe_str = str(row.get('amount_safe_to_pay', '')).strip()
            try:
                safe_val = float(safe_str)
                if safe_val < -1e-4 or safe_val > req_amt + 1e-4:
                    errors.append(f"{row_prefix} amount_safe_to_pay {safe_val} out of bounds [0, {req_amt}]")
            except ValueError:
                errors.append(f"{row_prefix} Invalid amount_safe_to_pay: '{safe_str}'")

            # 4. affordability_status
            status = str(row.get('affordability_status', '')).strip()
            if status not in VALID_AFFORDABILITY_STATUSES:
                errors.append(f"{row_prefix} Invalid affordability_status: '{status}'")

            # 5. recommended_payment_method
            method = str(row.get('recommended_payment_method', '')).strip()
            if method not in VALID_PAYMENT_METHODS:
                errors.append(f"{row_prefix} Invalid recommended_payment_method: '{method}'")

            # 6. payment_plan
            plan = str(row.get('payment_plan', '')).strip()
            if method == 'not_recommended':
                if plan != 'none':
                    errors.append(f"{row_prefix} payment_plan must be 'none' for not_recommended, got '{plan}'")
            else:
                if plan == 'none' or not PLAN_REGEX.match(plan):
                    errors.append(f"{row_prefix} Invalid payment_plan format: '{plan}'")

            # 7. earliest_date_for_full_payment
            earliest_d = str(row.get('earliest_date_for_full_payment', '')).strip()
            if status == 'affordable_now':
                if earliest_d != req_date:
                    errors.append(f"{row_prefix} For affordable_now, earliest_date must equal request_date {req_date}, got '{earliest_d}'")
            elif earliest_d != '':
                if not DATE_REGEX.match(earliest_d):
                    errors.append(f"{row_prefix} Invalid earliest_date_for_full_payment format: '{earliest_d}'")

            # 8. spending_changes_needed
            changes = str(row.get('spending_changes_needed', '')).strip()
            if changes != 'none':
                parts = changes.split('|')
                if len(parts) > 3:
                    errors.append(f"{row_prefix} More than 3 spending changes: {len(parts)}")
                for p in parts:
                    if not SPENDING_CHANGE_REGEX.match(p):
                        errors.append(f"{row_prefix} Invalid spending change syntax: '{p}'")

            # 9. decision_explanation
            explanation = str(row.get('decision_explanation', '')).strip()
            if not explanation or len(explanation) < 10:
                errors.append(f"{row_prefix} Missing or too short decision_explanation")

        is_valid = (len(errors) == 0)
        return is_valid, errors
