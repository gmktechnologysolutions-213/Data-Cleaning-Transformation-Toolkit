from dataclasses import dataclass, field
from typing import List, Optional, Dict, Any
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, MinMaxScaler, RobustScaler, OrdinalEncoder
from .utils.logging import get_logger

def _infer_cats(df: pd.DataFrame) -> List[str]:
    return [c for c in df.columns if df[c].dtype == "object" or str(df[c].dtype).startswith("category")]

def _infer_nums(df: pd.DataFrame) -> List[str]:
    return [c for c in df.columns if pd.api.types.is_numeric_dtype(df[c])]

@dataclass
class remove_duplicates:
    subset: Optional[List[str]] = None
    keep: str = "first"
    log: Any = field(default_factory=lambda: get_logger("cleanit.remove_duplicates"))

    def fit(self, df: pd.DataFrame):
        return self

    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        before = len(df)
        out = df.drop_duplicates(subset=self.subset, keep=self.keep).reset_index(drop=True)
        after = len(out)
        self.log.info("Removed %d duplicates", before - after)
        return out

@dataclass
class fill_missing:
    num_strategy: str = "median"
    num_constant: float = 0.0
    cat_strategy: str = "mode"
    cat_constant: str = "missing"
    log: Any = field(default_factory=lambda: get_logger("cleanit.fill_missing"))

    def fit(self, df: pd.DataFrame):
        self.num_cols = _infer_nums(df)
        self.cat_cols = _infer_cats(df)
        self.num_stats: Dict[str, float] = {}
        for c in self.num_cols:
            if self.num_strategy == "median":
                self.num_stats[c] = float(df[c].median())
            elif self.num_strategy == "mean":
                self.num_stats[c] = float(df[c].mean())
            else:
                self.num_stats[c] = self.num_constant
        self.cat_stats: Dict[str, Any] = {}
        for c in self.cat_cols:
            if self.cat_strategy == "mode":
                self.cat_stats[c] = df[c].mode().iloc[0] if not df[c].mode().empty else self.cat_constant
            else:
                self.cat_stats[c] = self.cat_constant
        return self

    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        X = df.copy()
        for c, v in self.num_stats.items():
            X[c] = X[c].fillna(v)
        for c, v in self.cat_stats.items():
            X[c] = X[c].fillna(v)
        self.log.info("Filled missing values in %d numeric and %d categorical columns", len(self.num_stats), len(self.cat_stats))
        return X

@dataclass
class encode_categoricals:
    method: str = "onehot"   # onehot | ordinal
    handle_unknown: str = "ignore"
    log: Any = field(default_factory=lambda: get_logger("cleanit.encode"))

    def fit(self, df: pd.DataFrame):
        self.cat_cols = _infer_cats(df)
        if self.method == "ordinal":
            self.encoder = OrdinalEncoder(handle_unknown="use_encoded_value", unknown_value=-1)
            self.encoder.fit(df[self.cat_cols] if self.cat_cols else pd.DataFrame())
        else:
            self.encoder = None
        return self

    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        X = df.copy()
        cats = [c for c in self.cat_cols if c in X.columns]
        if not cats:
            return X
        if self.method == "ordinal":
            X[cats] = self.encoder.transform(X[cats])
            return X
        X = pd.get_dummies(X, columns=cats, drop_first=False)
        return X

@dataclass
class scale_features:
    method: str = "standard"
    log: Any = field(default_factory=lambda: get_logger("cleanit.scale"))

    def fit(self, df: pd.DataFrame):
        self.num_cols = _infer_nums(df)
        if self.method == "standard":
            self.scaler = StandardScaler()
        elif self.method == "minmax":
            self.scaler = MinMaxScaler()
        else:
            self.scaler = RobustScaler()
        if self.num_cols:
            self.scaler.fit(df[self.num_cols].astype(float).values)
        return self

    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        X = df.copy()
        if self.num_cols:
            X[self.num_cols] = self.scaler.transform(X[self.num_cols].astype(float).values)
        self.log.info("Scaled %d numeric columns using %s", len(self.num_cols), self.method)
        return X

def report_profile(df: pd.DataFrame) -> pd.DataFrame:
    cols = df.columns
    report = []
    for c in cols:
        miss = df[c].isna().mean() * 100
        distinct = df[c].nunique()
        dtype = str(df[c].dtype)
        row = {"column": c, "dtype": dtype, "missing_%": round(miss, 2), "distinct": int(distinct)}
        if pd.api.types.is_numeric_dtype(df[c]):
            row.update({
                "mean": float(np.nanmean(df[c])) if df[c].notna().any() else None,
                "std": float(np.nanstd(df[c])) if df[c].notna().any() else None,
                "min": float(np.nanmin(df[c])) if df[c].notna().any() else None,
                "max": float(np.nanmax(df[c])) if df[c].notna().any() else None,
            })
        report.append(row)
    return pd.DataFrame(report)
