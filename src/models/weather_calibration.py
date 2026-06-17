import numpy as np
import pandas as pd
from typing import Optional, Dict, Any


def build_fast_calibration_sample(df: pd.DataFrame, max_rows: int = 200) -> pd.DataFrame:
    if df is None or df.empty:
        return pd.DataFrame()

    working = df.copy()

    needed = [
        "enhanced_model_probability",
        "implied_probability",
        "best_ev",
        "bet_side",
        "market_family",
        "model_supported",
    ]
    for col in needed:
        if col not in working.columns:
            working[col] = np.nan

    working = working[
        (working["model_supported"] == True) &
        (working["market_family"] == "temperature") &
        working["enhanced_model_probability"].notna() &
        working["implied_probability"].notna()
    ].copy()

    if working.empty:
        return pd.DataFrame()

    working["confidence"] = np.maximum(
        working["enhanced_model_probability"],
        1 - working["enhanced_model_probability"]
    )

    working = working.sort_values(
        ["confidence", "best_ev"],
        ascending=[False, False]
    ).head(max_rows)

    return working


def conservative_probability_shrink(
    p: float,
    market_p: float,
    shrink_to_market: float = 0.60,
    shrink_to_coinflip: float = 0.15,
    min_prob: float = 0.02,
    max_prob: float = 0.98,
) -> float:
    if p is None or pd.isna(p):
        return np.nan
    if market_p is None or pd.isna(market_p):
        market_p = 0.5

    calibrated = (
        (1 - shrink_to_market - shrink_to_coinflip) * float(p)
        + shrink_to_market * float(market_p)
        + shrink_to_coinflip * 0.5
    )

    return float(np.clip(calibrated, min_prob, max_prob))


def estimate_calibration_metrics(df_sample: pd.DataFrame) -> Optional[Dict[str, float]]:
    if df_sample is None or df_sample.empty:
        return None

    raw_gap = np.abs(
        df_sample["enhanced_model_probability"] - df_sample["implied_probability"]
    )

    mean_gap = float(raw_gap.mean()) if len(raw_gap) else 0.0

    pseudo_log_loss = float(min(max(mean_gap + 0.45, 0.45), 0.95))
    pseudo_brier = float(min(max((mean_gap ** 2) + 0.08, 0.08), 0.35))

    return {
        "log_loss": pseudo_log_loss,
        "brier_score": pseudo_brier,
        "sample_size": int(len(df_sample)),
        "mode": "conservative_shrinkage",
    }


def cap_kelly_fraction(
    raw_kelly: float,
    calibration_metrics: Optional[Dict[str, float]],
    max_kelly: float = 0.05,
    soft_cap_threshold_logloss: float = 0.60,
):
    if raw_kelly is None or pd.isna(raw_kelly):
        return 0.0

    raw_kelly = max(float(raw_kelly), 0.0)

    if calibration_metrics is None:
        return min(raw_kelly, 0.02)

    logloss = float(calibration_metrics.get("log_loss", 1.0))

    if logloss >= 0.80:
        allowed = 0.01
    elif logloss >= 0.70:
        allowed = 0.015
    elif logloss >= soft_cap_threshold_logloss:
        allowed = 0.025
    else:
        allowed = max_kelly

    return min(raw_kelly, allowed)


def compute_sane_recommended_bet(
    bankroll: float,
    raw_kelly_bet: float,
    edge: float,
    calibrated_prob: float,
    calibration_metrics: Optional[Dict[str, float]],
    min_bet_floor: float = 5.0,
    max_bet_floor: float = 25.0,
    max_edge_to_bet: float = 0.15,
):
    bankroll = float(bankroll) if bankroll is not None else 0.0
    raw_kelly_bet = float(raw_kelly_bet) if raw_kelly_bet is not None else 0.0
    edge = float(edge) if edge is not None and not pd.isna(edge) else 0.0
    calibrated_prob = float(calibrated_prob) if calibrated_prob is not None and not pd.isna(calibrated_prob) else 0.5

    if bankroll <= 0:
        return 0.0

    raw_kelly_fraction = raw_kelly_bet / bankroll
    capped_kelly_fraction = cap_kelly_fraction(
        raw_kelly_fraction,
        calibration_metrics,
        max_kelly=0.05,
        soft_cap_threshold_logloss=0.60,
    )

    capped_bet = bankroll * capped_kelly_fraction

    if edge > max_edge_to_bet:
        capped_bet = min(capped_bet, max_bet_floor)

    if calibrated_prob < 0.55:
        capped_bet = min(capped_bet, min_bet_floor)

    if calibration_metrics is None:
        min_bet = min_bet_floor
    else:
        logloss = float(calibration_metrics.get("log_loss", 1.0))
        if logloss >= 0.80:
            min_bet = min_bet_floor
        elif logloss >= 0.65:
            min_bet = min(min_bet_floor * 1.0, max_bet_floor)
        else:
            min_bet = min(min_bet_floor * 1.5, max_bet_floor)

    final_bet = max(capped_bet, min_bet)
    final_bet = min(final_bet, max_bet_floor, bankroll * 0.05)

    return float(final_bet)


def run_calibration_on_dashboard_df(df: pd.DataFrame):
    if df is None or df.empty:
        return df, None, None

    out = df.copy()

    if "enhanced_model_probability" not in out.columns:
        out["calibrated_model_probability"] = np.nan
        out["calibrated_yes_edge"] = np.nan
        out["calibrated_no_edge"] = np.nan
        out["calibrated_best_ev"] = np.nan
        return out, None, None

    sample = build_fast_calibration_sample(out, max_rows=200)
    metrics = estimate_calibration_metrics(sample)

    out["calibrated_model_probability"] = out.apply(
        lambda row: conservative_probability_shrink(
            row.get("enhanced_model_probability"),
            row.get("implied_probability"),
            shrink_to_market=0.60,
            shrink_to_coinflip=0.15,
            min_prob=0.02,
            max_prob=0.98,
        ),
        axis=1,
    )

    out["calibrated_yes_edge"] = (
        out["calibrated_model_probability"] - out["implied_probability"]
    )

    out["calibrated_no_edge"] = (
        (1 - out["calibrated_model_probability"]) - (1 - out["implied_probability"])
    )

    out["calibrated_best_ev"] = np.select(
        [
            out["bet_side"] == "YES",
            out["bet_side"] == "NO",
        ],
        [
            out["calibrated_yes_edge"],
            out["calibrated_no_edge"],
        ],
        default=np.nan,
    )

    return out, None, metrics