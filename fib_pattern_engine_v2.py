
"""
Fib Pattern Engine V2
---------------------
Engine analisis setup trading berbasis pola historis untuk 3 keluaran utama:
1) First-hit probability
2) Reach probability semua fib dalam 48 jam
3) Continuation probability antar level fib searah

Output cocok untuk kebutuhan evaluasi setup:
- target pertama yang paling mungkin
- semua probabilitas fib up/down
- peluang lanjut 1.61 -> 2.5 -> 3.6
- risk TIE_SAME_BAR / NO_HIT_48H
- history kasus paling mirip (tanggal & jam)

Fitur input:
- Trend
- Squeeze Momentum
- Squeeze Momentum2
- Bar 1
- Bar 2
- Raw Position
- Final Position
- Score
- Last TR

Label/target yang dibaca dari dataset:
- marker hit fib (✅/❌) dan/atau rank fib
- rank fib dipakai untuk first-hit
- marker/rank dipakai untuk reach
"""
from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import joblib
import numpy as np
import pandas as pd
from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import OneHotEncoder, StandardScaler


FEATURE_CATEGORICAL = [
    "Trend",
    "Squeeze Momentum",
    "Squeeze Momentum2",
    "Bar 1",
    "Bar 2",
    "Raw Position",
    "Final Position",
]

FEATURE_NUMERIC = [
    "Score",
    "Last TR",
]

META_COLUMNS = ["Date", "Clock"]

RANK_COLUMNS = {
    "1.61_UP": "Fib 1,61 Up2",
    "1.61_DOWN": "Fib 1,61 Down3",
    "2.5_UP": "Fib 2,5 Up4",
    "2.5_DOWN": "Fib 2,5 Down5",
    "3.6_UP": "Fib 3,6 Up6",
    "3.6_DOWN": "Fib 3,6 Down7",
}

HIT_COLUMNS = {
    "1.61_UP": "Fib 1,61 Up",
    "1.61_DOWN": "Fib 1,61 Down",
    "2.5_UP": "Fib 2,5 Up",
    "2.5_DOWN": "Fib 2,5 Down",
    "3.6_UP": "Fib 3,6 Up",
    "3.6_DOWN": "Fib 3,6 Down",
}

ACTIONABLE_TARGETS = list(RANK_COLUMNS.keys())
ALL_FIRST_HIT_TARGETS = ACTIONABLE_TARGETS + ["TIE_SAME_BAR", "NO_HIT_48H"]

CONTINUATION_KEYS = [
    "UP_1.61_TO_2.5",
    "UP_2.5_TO_3.6",
    "UP_1.61_TO_3.6",
    "DOWN_1.61_TO_2.5",
    "DOWN_2.5_TO_3.6",
    "DOWN_1.61_TO_3.6",
]

MONTH_MAP_ID = {
    "januari": 1,
    "februari": 2,
    "maret": 3,
    "april": 4,
    "mei": 5,
    "juni": 6,
    "juli": 7,
    "agustus": 8,
    "september": 9,
    "oktober": 10,
    "november": 11,
    "desember": 12,
}


@dataclass
class PredictionResultV2:
    first_hit_top_target: Optional[str]
    first_hit_top_prob: float
    first_hit_second_target: Optional[str]
    first_hit_second_prob: float
    first_hit_probs: Dict[str, float]

    tie_prob: float
    no_hit_prob: float

    reach_top_target: Optional[str]
    reach_top_prob: float
    reach_second_target: Optional[str]
    reach_second_prob: float
    reach_probs: Dict[str, float]

    continuation_probs: Dict[str, float]
    source_summary: Dict[str, float]
    top_matches: List[Dict[str, object]]


def _make_one_hot_encoder():
    try:
        return OneHotEncoder(handle_unknown="ignore", sparse_output=False)
    except TypeError:
        return OneHotEncoder(handle_unknown="ignore", sparse=False)


class FibPatternEngineV2:
    def __init__(
        self,
        exact_count_smoothing: int = 5,
        similarity_k: int = 10,
        numeric_weight_score: float = 5.0,
        numeric_weight_last_tr: float = 1.0,
        continuation_smoothing: float = 1.0,
    ) -> None:
        self.exact_count_smoothing = exact_count_smoothing
        self.similarity_k = similarity_k
        self.numeric_weight_score = numeric_weight_score
        self.numeric_weight_last_tr = numeric_weight_last_tr
        self.continuation_smoothing = continuation_smoothing

        self.raw_df: Optional[pd.DataFrame] = None
        self.train_df: Optional[pd.DataFrame] = None

        self.encoder: Optional[OneHotEncoder] = None
        self.scaler: Optional[StandardScaler] = None
        self.nn_model: Optional[NearestNeighbors] = None
        self.feature_matrix: Optional[np.ndarray] = None

        self.first_hit_pattern_store: Dict[Tuple, Dict[str, float]] = {}
        self.reach_pattern_store: Dict[Tuple, Dict[str, float]] = {}
        self.continuation_pattern_store: Dict[Tuple, Dict[str, float]] = {}
        self.pattern_counts: Dict[Tuple, int] = {}

        self.global_first_hit_probs: Dict[str, float] = {}
        self.global_reach_probs: Dict[str, float] = {}
        self.global_continuation_probs: Dict[str, float] = {}

    # =====================
    # PUBLIC API
    # =====================
    def fit(self, excel_path: str | Path, sheet_name: str | int = 0) -> "FibPatternEngineV2":
        df = pd.read_excel(excel_path, sheet_name=sheet_name)
        df = self._prepare_dataframe(df)

        if df.empty:
            raise ValueError("Tidak ada baris valid yang bisa dipakai untuk training.")

        self.raw_df = df.copy()
        self.train_df = df.copy()

        self._build_pattern_stores(df)
        self._build_similarity_index(df)

        self.global_first_hit_probs = self._value_counts_to_probs(df["first_hit_target"].value_counts(), ALL_FIRST_HIT_TARGETS)
        self.global_reach_probs = self._mean_probs(df, ACTIONABLE_TARGETS, prefix="reach_")
        self.global_continuation_probs = self._compute_continuation_probs(df)

        return self

    def save(self, output_path: str | Path) -> None:
        payload = {
            "exact_count_smoothing": self.exact_count_smoothing,
            "similarity_k": self.similarity_k,
            "numeric_weight_score": self.numeric_weight_score,
            "numeric_weight_last_tr": self.numeric_weight_last_tr,
            "continuation_smoothing": self.continuation_smoothing,
            "raw_df": self.raw_df,
            "train_df": self.train_df,
            "encoder": self.encoder,
            "scaler": self.scaler,
            "nn_model": self.nn_model,
            "feature_matrix": self.feature_matrix,
            "first_hit_pattern_store": self.first_hit_pattern_store,
            "reach_pattern_store": self.reach_pattern_store,
            "continuation_pattern_store": self.continuation_pattern_store,
            "pattern_counts": self.pattern_counts,
            "global_first_hit_probs": self.global_first_hit_probs,
            "global_reach_probs": self.global_reach_probs,
            "global_continuation_probs": self.global_continuation_probs,
        }
        joblib.dump(payload, output_path)

    @classmethod
    def load(cls, model_path: str | Path) -> "FibPatternEngineV2":
        payload = joblib.load(model_path)
        engine = cls(
            exact_count_smoothing=payload["exact_count_smoothing"],
            similarity_k=payload["similarity_k"],
            numeric_weight_score=payload["numeric_weight_score"],
            numeric_weight_last_tr=payload["numeric_weight_last_tr"],
            continuation_smoothing=payload.get("continuation_smoothing", 1.0),
        )
        engine.raw_df = payload["raw_df"]
        engine.train_df = payload["train_df"]
        engine.encoder = payload["encoder"]
        engine.scaler = payload["scaler"]
        engine.nn_model = payload["nn_model"]
        engine.feature_matrix = payload["feature_matrix"]
        engine.first_hit_pattern_store = payload["first_hit_pattern_store"]
        engine.reach_pattern_store = payload["reach_pattern_store"]
        engine.continuation_pattern_store = payload["continuation_pattern_store"]
        engine.pattern_counts = payload["pattern_counts"]
        engine.global_first_hit_probs = payload["global_first_hit_probs"]
        engine.global_reach_probs = payload["global_reach_probs"]
        engine.global_continuation_probs = payload["global_continuation_probs"]
        return engine

    def predict(self, setup: Dict[str, object], top_k_matches: int = 5) -> PredictionResultV2:
        self._assert_is_fitted()
        row = self._normalize_single_setup(setup)

        exact_count = self.pattern_counts.get(tuple(row[FEATURE_CATEGORICAL + ["Score"]].iloc[0].values.tolist()), 0)
        exact_weight = self._exact_weight(exact_count)

        # exact stores
        exact_first_hit = self.first_hit_pattern_store.get(tuple(row[FEATURE_CATEGORICAL + ["Score"]].iloc[0].values.tolist()), self.global_first_hit_probs)
        exact_reach = self.reach_pattern_store.get(tuple(row[FEATURE_CATEGORICAL + ["Score"]].iloc[0].values.tolist()), self.global_reach_probs)
        exact_cont = self.continuation_pattern_store.get(tuple(row[FEATURE_CATEGORICAL + ["Score"]].iloc[0].values.tolist()), self.global_continuation_probs)

        # similarity stores
        sim_first_hit, sim_reach, sim_cont, top_matches = self._predict_from_similarity(row, top_k_matches=top_k_matches)

        # blend
        blended_first_hit = self._blend_multiclass_probs(exact_first_hit, sim_first_hit, exact_weight, ALL_FIRST_HIT_TARGETS)
        blended_reach = self._blend_binary_probs(exact_reach, sim_reach, exact_weight, ACTIONABLE_TARGETS)
        blended_cont = self._blend_binary_probs(exact_cont, sim_cont, exact_weight, CONTINUATION_KEYS)

        fib_first_hit_only = {k: blended_first_hit.get(k, 0.0) for k in ACTIONABLE_TARGETS}
        first_top_target, first_top_prob = self._top_from_probs(fib_first_hit_only, rank=1)
        first_second_target, first_second_prob = self._top_from_probs(fib_first_hit_only, rank=2)

        reach_top_target, reach_top_prob = self._top_from_probs(blended_reach, rank=1)
        reach_second_target, reach_second_prob = self._top_from_probs(blended_reach, rank=2)

        source_summary = {
            "exact_match_count": float(exact_count),
            "exact_weight_used": exact_weight,
            "similarity_weight_used": 1.0 - exact_weight,
        }

        return PredictionResultV2(
            first_hit_top_target=first_top_target,
            first_hit_top_prob=first_top_prob,
            first_hit_second_target=first_second_target,
            first_hit_second_prob=first_second_prob,
            first_hit_probs=blended_first_hit,
            tie_prob=blended_first_hit.get("TIE_SAME_BAR", 0.0),
            no_hit_prob=blended_first_hit.get("NO_HIT_48H", 0.0),
            reach_top_target=reach_top_target,
            reach_top_prob=reach_top_prob,
            reach_second_target=reach_second_target,
            reach_second_prob=reach_second_prob,
            reach_probs=blended_reach,
            continuation_probs=blended_cont,
            source_summary=source_summary,
            top_matches=top_matches,
        )

    def summarize_first_hit_patterns(self) -> pd.DataFrame:
        self._assert_is_fitted()
        rows = []
        for key, probs in self.first_hit_pattern_store.items():
            row = dict(zip(FEATURE_CATEGORICAL + ["Score"], key))
            row["match_count"] = self.pattern_counts[key]
            for target in ALL_FIRST_HIT_TARGETS:
                row[f"p_first_{target}"] = probs.get(target, 0.0)
            rows.append(row)
        out = pd.DataFrame(rows)
        if not out.empty:
            out = out.sort_values(by="match_count", ascending=False).reset_index(drop=True)
        return out

    def summarize_reach_patterns(self) -> pd.DataFrame:
        self._assert_is_fitted()
        rows = []
        for key, probs in self.reach_pattern_store.items():
            row = dict(zip(FEATURE_CATEGORICAL + ["Score"], key))
            row["match_count"] = self.pattern_counts[key]
            for target in ACTIONABLE_TARGETS:
                row[f"p_reach_{target}"] = probs.get(target, 0.0)
            for ckey in CONTINUATION_KEYS:
                row[f"p_cont_{ckey}"] = self.continuation_pattern_store.get(key, {}).get(ckey, np.nan)
            rows.append(row)
        out = pd.DataFrame(rows)
        if not out.empty:
            out = out.sort_values(by="match_count", ascending=False).reset_index(drop=True)
        return out

    # =====================
    # PREP
    # =====================
    def _prepare_dataframe(self, df: pd.DataFrame) -> pd.DataFrame:
        df = df.copy()

        required_columns = FEATURE_CATEGORICAL + FEATURE_NUMERIC + META_COLUMNS + list(RANK_COLUMNS.values())
        missing = [col for col in required_columns if col not in df.columns]
        if missing:
            raise ValueError(f"Kolom wajib tidak ditemukan di Excel: {missing}")

        keep_cols = required_columns + [c for c in HIT_COLUMNS.values() if c in df.columns]
        df = df[keep_cols].copy()

        for col in FEATURE_CATEGORICAL:
            df[col] = df[col].apply(self._normalize_text)

        df["Score"] = pd.to_numeric(df["Score"], errors="coerce")
        df["Last TR"] = df["Last TR"].apply(self._parse_local_number)

        df["Date"] = df["Date"].apply(self._parse_indonesian_date)
        df["Clock"] = pd.to_numeric(df["Clock"], errors="coerce")

        for target, rank_col in RANK_COLUMNS.items():
            df[rank_col] = pd.to_numeric(df[rank_col], errors="coerce").fillna(0).astype(int)

        for target, hit_col in HIT_COLUMNS.items():
            if hit_col in df.columns:
                df[f"hit_marker_{target}"] = df[hit_col].apply(self._parse_hit_marker)
            else:
                df[f"hit_marker_{target}"] = False

        # Valid training rows must have all features + meta
        mask_valid = df[FEATURE_CATEGORICAL + FEATURE_NUMERIC + META_COLUMNS].notna().all(axis=1)
        df = df.loc[mask_valid].copy()

        if df.empty:
            return df

        # reach flags
        for target in ACTIONABLE_TARGETS:
            rank_col = RANK_COLUMNS[target]
            marker_col = f"hit_marker_{target}"
            df[f"reach_{target}"] = (df[rank_col] > 0) | (df[marker_col].fillna(False))

        df["first_hit_target"] = df.apply(self._derive_first_hit_target, axis=1)
        df["first_hit_direction"] = df["first_hit_target"].apply(self._first_hit_to_direction)
        df["first_hit_level"] = df["first_hit_target"].apply(self._first_hit_to_level)
        df["reached_targets"] = df.apply(self._make_reached_targets_label, axis=1)
        df["timestamp_label"] = df.apply(self._make_timestamp_label, axis=1)

        exact_key_cols = FEATURE_CATEGORICAL + ["Score"]
        df["exact_key"] = df[exact_key_cols].apply(lambda s: tuple(s.values.tolist()), axis=1)

        return df.reset_index(drop=True)

    def _build_pattern_stores(self, df: pd.DataFrame) -> None:
        self.first_hit_pattern_store = {}
        self.reach_pattern_store = {}
        self.continuation_pattern_store = {}
        self.pattern_counts = {}

        grouped = df.groupby("exact_key")

        for key, sub in grouped:
            self.pattern_counts[key] = int(len(sub))
            self.first_hit_pattern_store[key] = self._value_counts_to_probs(sub["first_hit_target"].value_counts(), ALL_FIRST_HIT_TARGETS)
            self.reach_pattern_store[key] = self._mean_probs(sub, ACTIONABLE_TARGETS, prefix="reach_")
            self.continuation_pattern_store[key] = self._compute_continuation_probs(sub)

    def _build_similarity_index(self, df: pd.DataFrame) -> None:
        self.encoder = _make_one_hot_encoder()
        X_cat = self.encoder.fit_transform(df[FEATURE_CATEGORICAL])

        self.scaler = StandardScaler()
        X_num = self.scaler.fit_transform(df[FEATURE_NUMERIC].astype(float).copy())
        X_num[:, 0] = X_num[:, 0] * self.numeric_weight_score
        X_num[:, 1] = X_num[:, 1] * self.numeric_weight_last_tr

        self.feature_matrix = np.hstack([X_cat, X_num])

        n_neighbors = min(self.similarity_k, len(df))
        self.nn_model = NearestNeighbors(n_neighbors=n_neighbors, metric="euclidean")
        self.nn_model.fit(self.feature_matrix)

    # =====================
    # SIMILARITY PREDICT
    # =====================
    def _predict_from_similarity(self, row: pd.DataFrame, top_k_matches: int = 5):
        X_query = self._transform_single_row(row)
        distances, indices = self.nn_model.kneighbors(X_query)
        distances = distances[0]
        indices = indices[0]

        eps = 1e-6
        weights = 1.0 / (distances + eps)
        if np.sum(weights) == 0:
            weights = np.ones_like(weights)
        weights = weights / np.sum(weights)

        first_hit_probs = {k: 0.0 for k in ALL_FIRST_HIT_TARGETS}
        reach_probs = {k: 0.0 for k in ACTIONABLE_TARGETS}
        top_matches = []

        sub_rows = []
        sub_weights = []

        for dist, idx, w in zip(distances, indices, weights):
            train_row = self.train_df.iloc[int(idx)]
            sub_rows.append(train_row)
            sub_weights.append(float(w))

            target = train_row["first_hit_target"]
            first_hit_probs[target] += float(w)

            for fib in ACTIONABLE_TARGETS:
                reach_probs[fib] += float(w) * float(bool(train_row[f"reach_{fib}"]))

            top_matches.append({
                "date": train_row.get("Date"),
                "clock": train_row.get("Clock"),
                "first_hit_target": train_row.get("first_hit_target"),
                "first_hit_direction": train_row.get("first_hit_direction"),
                "first_hit_level": train_row.get("first_hit_level"),
                "reached_targets": train_row.get("reached_targets"),
                "similarity": float(1.0 / (1.0 + dist)),
                "trend": train_row.get("Trend"),
                "score": train_row.get("Score"),
                "last_tr": train_row.get("Last TR"),
                "raw_position": train_row.get("Raw Position"),
                "final_position": train_row.get("Final Position"),
            })

        # normalize / clamp
        total_first = sum(first_hit_probs.values())
        if total_first > 0:
            first_hit_probs = {k: float(v / total_first) for k, v in first_hit_probs.items()}
        else:
            first_hit_probs = self.global_first_hit_probs.copy()

        reach_probs = {k: float(min(max(v, 0.0), 1.0)) for k, v in reach_probs.items()}
        cont_probs = self._compute_continuation_probs_weighted(sub_rows, sub_weights)

        top_matches = sorted(top_matches, key=lambda x: x["similarity"], reverse=True)[:top_k_matches]
        return first_hit_probs, reach_probs, cont_probs, top_matches

    # =====================
    # PRED HELPERS
    # =====================
    def _transform_single_row(self, row: pd.DataFrame) -> np.ndarray:
        X_cat = self.encoder.transform(row[FEATURE_CATEGORICAL])
        X_num = self.scaler.transform(row[FEATURE_NUMERIC].astype(float))
        X_num[:, 0] = X_num[:, 0] * self.numeric_weight_score
        X_num[:, 1] = X_num[:, 1] * self.numeric_weight_last_tr
        return np.hstack([X_cat, X_num])

    def _normalize_single_setup(self, setup: Dict[str, object]) -> pd.DataFrame:
        row = {}
        for col in FEATURE_CATEGORICAL:
            if col not in setup:
                raise ValueError(f"Input setup kurang kolom: {col}")
            row[col] = self._normalize_text(setup[col])

        if "Score" not in setup or "Last TR" not in setup:
            raise ValueError("Input setup wajib punya Score dan Last TR.")

        row["Score"] = float(setup["Score"])
        row["Last TR"] = float(setup["Last TR"])
        row["Date"] = None
        row["Clock"] = None
        return pd.DataFrame([row])

    # =====================
    # TARGET ENGINEERING
    # =====================
    def _derive_first_hit_target(self, row: pd.Series) -> str:
        ranks = {target: int(row[rank_col]) for target, rank_col in RANK_COLUMNS.items()}
        positive = {target: rank for target, rank in ranks.items() if rank > 0}

        if not positive:
            return "NO_HIT_48H"

        min_rank = min(positive.values())
        winners = [target for target, rank in positive.items() if rank == min_rank]

        if len(winners) > 1:
            return "TIE_SAME_BAR"
        return winners[0]

    @staticmethod
    def _first_hit_to_direction(target: str) -> Optional[str]:
        if target.endswith("_UP"):
            return "UP_FIRST"
        if target.endswith("_DOWN"):
            return "DOWN_FIRST"
        return None

    @staticmethod
    def _first_hit_to_level(target: str) -> Optional[str]:
        match = re.match(r"^(1\.61|2\.5|3\.6)_(UP|DOWN)$", target)
        if match:
            return match.group(1)
        return None

    def _compute_continuation_probs(self, df: pd.DataFrame) -> Dict[str, float]:
        out = {}
        # up side
        out["UP_1.61_TO_2.5"] = self._conditional_prob(df, "reach_1.61_UP", "reach_2.5_UP")
        out["UP_2.5_TO_3.6"] = self._conditional_prob(df, "reach_2.5_UP", "reach_3.6_UP")
        out["UP_1.61_TO_3.6"] = self._conditional_prob(df, "reach_1.61_UP", "reach_3.6_UP")
        # down side
        out["DOWN_1.61_TO_2.5"] = self._conditional_prob(df, "reach_1.61_DOWN", "reach_2.5_DOWN")
        out["DOWN_2.5_TO_3.6"] = self._conditional_prob(df, "reach_2.5_DOWN", "reach_3.6_DOWN")
        out["DOWN_1.61_TO_3.6"] = self._conditional_prob(df, "reach_1.61_DOWN", "reach_3.6_DOWN")
        return out

    def _compute_continuation_probs_weighted(self, rows: List[pd.Series], weights: List[float]) -> Dict[str, float]:
        if not rows:
            return self.global_continuation_probs.copy()

        arr = pd.DataFrame(rows).reset_index(drop=True)
        w = np.array(weights, dtype=float)
        if w.sum() <= 0:
            w = np.ones(len(arr), dtype=float) / max(len(arr), 1)
        else:
            w = w / w.sum()

        out = {}
        out["UP_1.61_TO_2.5"] = self._conditional_prob_weighted(arr, w, "reach_1.61_UP", "reach_2.5_UP")
        out["UP_2.5_TO_3.6"] = self._conditional_prob_weighted(arr, w, "reach_2.5_UP", "reach_3.6_UP")
        out["UP_1.61_TO_3.6"] = self._conditional_prob_weighted(arr, w, "reach_1.61_UP", "reach_3.6_UP")
        out["DOWN_1.61_TO_2.5"] = self._conditional_prob_weighted(arr, w, "reach_1.61_DOWN", "reach_2.5_DOWN")
        out["DOWN_2.5_TO_3.6"] = self._conditional_prob_weighted(arr, w, "reach_2.5_DOWN", "reach_3.6_DOWN")
        out["DOWN_1.61_TO_3.6"] = self._conditional_prob_weighted(arr, w, "reach_1.61_DOWN", "reach_3.6_DOWN")
        return out

    def _conditional_prob(self, df: pd.DataFrame, given_col: str, target_col: str) -> float:
        given = df[given_col].astype(bool)
        target = df[target_col].astype(bool)

        denom = int(given.sum())
        numer = int((given & target).sum())

        alpha = float(self.continuation_smoothing)
        # Beta(1,1)-style smoothing
        return float((numer + alpha) / (denom + 2 * alpha)) if denom >= 0 else 0.5

    def _conditional_prob_weighted(self, df: pd.DataFrame, weights: np.ndarray, given_col: str, target_col: str) -> float:
        given = df[given_col].astype(bool).values
        target = df[target_col].astype(bool).values

        denom = float(weights[given].sum())
        numer = float(weights[given & target].sum())

        alpha = float(self.continuation_smoothing)
        return float((numer + alpha) / (denom + 2 * alpha))

    def _mean_probs(self, df: pd.DataFrame, targets: List[str], prefix: str) -> Dict[str, float]:
        out = {}
        for t in targets:
            col = f"{prefix}{t}"
            out[t] = float(df[col].astype(float).mean()) if col in df.columns else 0.0
        return out

    def _blend_multiclass_probs(
        self,
        exact_probs: Dict[str, float],
        sim_probs: Dict[str, float],
        exact_weight: float,
        targets: List[str],
    ) -> Dict[str, float]:
        out = {}
        sim_weight = 1.0 - exact_weight
        for t in targets:
            out[t] = float(exact_weight * exact_probs.get(t, 0.0) + sim_weight * sim_probs.get(t, 0.0))

        total = sum(out.values())
        if total > 0:
            out = {k: float(v / total) for k, v in out.items()}
        else:
            base = 1.0 / max(len(targets), 1)
            out = {k: base for k in targets}
        return out

    def _blend_binary_probs(
        self,
        exact_probs: Dict[str, float],
        sim_probs: Dict[str, float],
        exact_weight: float,
        targets: List[str],
    ) -> Dict[str, float]:
        sim_weight = 1.0 - exact_weight
        out = {}
        for t in targets:
            v = exact_weight * exact_probs.get(t, 0.0) + sim_weight * sim_probs.get(t, 0.0)
            out[t] = float(min(max(v, 0.0), 1.0))
        return out

    def _exact_weight(self, exact_count: int) -> float:
        if exact_count <= 0:
            return 0.0
        return float(exact_count / (exact_count + self.exact_count_smoothing))

    # =====================
    # SMALL HELPERS
    # =====================
    @staticmethod
    def _normalize_text(value: object) -> Optional[str]:
        if pd.isna(value):
            return None
        text = str(value).strip()
        if text == "":
            return None
        return text

    @staticmethod
    def _parse_local_number(value: object) -> Optional[float]:
        if pd.isna(value):
            return None
        if isinstance(value, (int, float, np.integer, np.floating)):
            return float(value)
        text = str(value).strip()
        if text == "":
            return None
        text = text.replace(".", "").replace(",", ".")
        try:
            return float(text)
        except ValueError:
            return None

    @staticmethod
    def _parse_hit_marker(value: object) -> bool:
        if pd.isna(value):
            return False
        if isinstance(value, (int, float, np.integer, np.floating)):
            return float(value) > 0
        text = str(value).strip().lower()
        if text in {"✅", "check", "checked", "true", "yes", "y", "1", "hit"}:
            return True
        if text in {"❌", "x", "false", "no", "n", "0", ""}:
            return False
        return "✅" in text or "hit" in text

    @staticmethod
    def _parse_indonesian_date(value: object) -> Optional[pd.Timestamp]:
        if pd.isna(value):
            return None
        if isinstance(value, pd.Timestamp):
            return value.normalize()

        text = str(value).strip()
        if text == "":
            return None

        parts = text.split()
        if len(parts) == 3:
            try:
                day = int(parts[0])
                month = MONTH_MAP_ID.get(parts[1].lower())
                year = int(parts[2])
                if month is not None:
                    return pd.Timestamp(year=year, month=month, day=day)
            except Exception:
                pass

        parsed = pd.to_datetime(text, errors="coerce")
        if pd.isna(parsed):
            return None
        return parsed.normalize()

    @staticmethod
    def _make_timestamp_label(row: pd.Series) -> Optional[str]:
        if pd.isna(row["Date"]) or pd.isna(row["Clock"]):
            return None
        dt = pd.Timestamp(row["Date"])
        hour = int(float(row["Clock"]))
        return f"{dt.strftime('%Y-%m-%d')} {hour:02d}:00"

    @staticmethod
    def _make_reached_targets_label(row: pd.Series) -> str:
        hits = [target for target in ACTIONABLE_TARGETS if bool(row.get(f"reach_{target}", False))]
        return ", ".join(hits) if hits else "NO_HIT_48H"

    @staticmethod
    def _value_counts_to_probs(counts: pd.Series, targets: List[str]) -> Dict[str, float]:
        out = {target: 0.0 for target in targets}
        total = float(counts.sum())
        if total <= 0:
            return out
        for target in targets:
            out[target] = float(counts.get(target, 0.0) / total)
        return out

    @staticmethod
    def _top_from_probs(probs: Dict[str, float], rank: int = 1) -> Tuple[Optional[str], float]:
        items = sorted(probs.items(), key=lambda x: x[1], reverse=True)
        if len(items) < rank:
            return None, 0.0
        key, value = items[rank - 1]
        return key, float(value)

    def _assert_is_fitted(self) -> None:
        if self.train_df is None or self.nn_model is None or self.encoder is None or self.scaler is None:
            raise RuntimeError("Engine belum di-fit. Jalankan .fit(...) dulu.")

    # =====================
    # REPORT / UTILITY
    # =====================
    def print_prediction(self, result: PredictionResultV2) -> None:
        print("=" * 84)
        print("HASIL ANALISIS SETUP - V2")
        print("=" * 84)
        print(f"First-hit target utama  : {result.first_hit_top_target} ({result.first_hit_top_prob:.2%})")
        print(f"First-hit kemungkinan 2 : {result.first_hit_second_target} ({result.first_hit_second_prob:.2%})")
        print(f"Risk TIE_SAME_BAR       : {result.tie_prob:.2%}")
        print(f"Risk NO_HIT_48H         : {result.no_hit_prob:.2%}")
        print()
        print(f"Reach target utama      : {result.reach_top_target} ({result.reach_top_prob:.2%})")
        print(f"Reach kemungkinan 2     : {result.reach_second_target} ({result.reach_second_prob:.2%})")
        print()
        print("Reach probability semua fib:")
        for key, val in sorted(result.reach_probs.items(), key=lambda x: x[1], reverse=True):
            print(f"  - {key:<12} : {val:.2%}")
        print()
        print("Continuation probability:")
        for key, val in result.continuation_probs.items():
            print(f"  - {key:<16} : {val:.2%}")
        print()
        print("Sumber keputusan:")
        for k, v in result.source_summary.items():
            if "weight" in k:
                print(f"  - {k:<22} : {v:.2%}")
            else:
                print(f"  - {k:<22} : {v:.0f}")
        print()
        print("Top historical matches:")
        for i, row in enumerate(result.top_matches, start=1):
            print(
                f"  {i}. {row.get('date')} | jam={row.get('clock')} | "
                f"first_hit={row.get('first_hit_target')} | reached=[{row.get('reached_targets')}] | "
                f"sim={row.get('similarity'):.4f} | trend={row.get('trend')} | "
                f"score={row.get('score')} | last_tr={row.get('last_tr')}"
            )
        print("=" * 84)


def train_and_save_model_v2(
    excel_path: str | Path,
    model_path: str | Path = "fib_pattern_engine_v2.pkl",
    first_hit_summary_csv: Optional[str | Path] = "fib_pattern_first_hit_summary_v2.csv",
    reach_summary_csv: Optional[str | Path] = "fib_pattern_reach_summary_v2.csv",
) -> FibPatternEngineV2:
    engine = FibPatternEngineV2()
    engine.fit(excel_path)
    engine.save(model_path)

    if first_hit_summary_csv is not None:
        engine.summarize_first_hit_patterns().to_csv(first_hit_summary_csv, index=False)
    if reach_summary_csv is not None:
        engine.summarize_reach_patterns().to_csv(reach_summary_csv, index=False)

    return engine


def example_manual_input() -> Dict[str, object]:
    return {
        "Trend": "Long",
        "Squeeze Momentum": "Rise weak white",
        "Squeeze Momentum2": "Rise strong white",
        "Bar 1": "Red Bar Line 5",
        "Bar 2": "Red Bar Line 2",
        "Raw Position": "Long",
        "Final Position": "Long",
        "Score": 5,
        "Last TR": 11.1,
    }


if __name__ == "__main__":
    excel_path = "Dataset Analisis Trading.xlsx"
    model_path = "fib_pattern_engine_v2.pkl"
    first_hit_summary = "fib_pattern_first_hit_summary_v2.csv"
    reach_summary = "fib_pattern_reach_summary_v2.csv"

    print("Training engine V2 dari Excel...")
    engine = train_and_save_model_v2(
        excel_path=excel_path,
        model_path=model_path,
        first_hit_summary_csv=first_hit_summary,
        reach_summary_csv=reach_summary,
    )
    print(f"Model tersimpan di: {model_path}")
    print(f"Summary first-hit tersimpan di: {first_hit_summary}")
    print(f"Summary reach tersimpan di: {reach_summary}")
    print()

    setup = example_manual_input()
    result = engine.predict(setup, top_k_matches=5)
    engine.print_prediction(result)
