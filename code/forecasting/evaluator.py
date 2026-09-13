"""
Regression Evaluator Module for Cash-Flow Forecasting.
Compares forecasting strategies against ground-truth sample requests.
Computes:
- MAE, RMSE, Pearson Correlation, Max Absolute Error (Safe Amount and Reserve)
- Per-request diagnostic tables containing all required audit fields
"""

from typing import Dict, List, Tuple
import numpy as np
import pandas as pd
from scipy.stats import pearsonr

from .normalization import EventNormalizer
from .strategies import ForecastingStrategy, SimulationResult


class ForecastingRegressionEvaluator:
    """Evaluates forecasting strategies against sample ground-truth."""

    def __init__(
        self,
        samples_df: pd.DataFrame,
        profiles_df: pd.DataFrame,
        normalizer: EventNormalizer,
    ):
        self.samples_df = samples_df.copy()
        self.profiles_df = profiles_df.copy()
        self.normalizer = normalizer

    def evaluate_strategy(
        self,
        strategy: ForecastingStrategy,
    ) -> Tuple[pd.DataFrame, Dict[str, float]]:
        """Run strategy over all sample requests and return per-request diagnostics and summary metrics."""
        diagnostics = []

        for _, sample in self.samples_df.iterrows():
            uid = sample['user_id']
            rid = sample['request_id']
            req_d = sample['request_date']
            req_amt = float(sample['requested_amount'])
            gt_safe = float(sample['amount_safe_to_pay'])

            prof = self.profiles_df[self.profiles_df['user_id'] == uid].iloc[0]
            curr_bal = float(prof['current_available_balance'])
            min_bal = float(prof['minimum_balance_to_keep'])
            home_curr = prof['home_currency']

            # Target reserve implied by ground-truth
            target_reserve = curr_bal - min_bal - gt_safe

            user_events = self.normalizer.get_user_events(uid, home_curr, req_d)

            # Predict
            sim_res: SimulationResult = strategy.predict(
                user_id=uid,
                user_events=user_events,
                current_available_balance=curr_bal,
                minimum_balance_to_keep=min_bal,
                requested_amount=req_amt,
                request_date=req_d,
            )

            err_safe = abs(sim_res.amount_safe_to_pay - gt_safe)
            err_res = abs(sim_res.predicted_reserve - target_reserve)

            diagnostics.append({
                'request_id': rid,
                'user_id': uid,
                'request_date': req_d,
                'current_balance': curr_bal,
                'minimum_balance': min_bal,
                'requested_amount': req_amt,
                'ground_truth_safe_amount': gt_safe,
                'target_reserve': target_reserve,
                'next_confirmed_income_date': sim_res.next_confirmed_income_date,
                'projected_minimum_balance_date': sim_res.projected_minimum_balance_date,
                'projected_minimum_balance': sim_res.projected_minimum_balance,
                'predicted_reserve': sim_res.predicted_reserve,
                'predicted_safe_amount': sim_res.amount_safe_to_pay,
                'error_safe_amount': err_safe,
                'error_reserve': err_res,
            })

        df_diag = pd.DataFrame(diagnostics)

        # Compute summary metrics
        mae_safe = float(df_diag['error_safe_amount'].mean())
        rmse_safe = float(np.sqrt((df_diag['error_safe_amount'] ** 2).mean()))
        corr_safe = float(pearsonr(df_diag['predicted_safe_amount'], df_diag['ground_truth_safe_amount'])[0])
        max_err_safe = float(df_diag['error_safe_amount'].max())

        mae_res = float(df_diag['error_reserve'].mean())
        rmse_res = float(np.sqrt((df_diag['error_reserve'] ** 2).mean()))
        corr_res = float(pearsonr(df_diag['predicted_reserve'], df_diag['target_reserve'])[0])
        max_err_res = float(df_diag['error_reserve'].max())

        norm_err = float((df_diag['error_safe_amount'] / np.maximum(df_diag['requested_amount'], 1.0)).mean() * 100)

        summary = {
            'strategy': strategy.name,
            'mae_safe': mae_safe,
            'rmse_safe': rmse_safe,
            'pearson_safe': corr_safe,
            'max_error_safe': max_err_safe,
            'norm_error_pct': norm_err,
            'mae_reserve': mae_res,
            'rmse_reserve': rmse_res,
            'pearson_reserve': corr_res,
            'max_error_reserve': max_err_res,
        }

        return df_diag, summary

    def run_suite(
        self,
        strategies: List[ForecastingStrategy],
    ) -> Tuple[Dict[str, pd.DataFrame], pd.DataFrame]:
        """Execute regression suite across all strategies."""
        all_diags = {}
        all_summaries = []

        for strat in strategies:
            diag, summ = self.evaluate_strategy(strat)
            all_diags[strat.name] = diag
            all_summaries.append(summ)

        summary_df = pd.DataFrame(all_summaries)
        return all_diags, summary_df
