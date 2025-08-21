import pickle
from typing import List, Tuple, Any
import pandas as pd
from .utils.logging import get_logger

class CleanPipeline:
    def __init__(self, steps: List[Any]):
        self.steps = steps
        self.fitted_steps: List[Tuple[str, Any]] = []
        self.log = get_logger("cleanit.pipeline")

    def fit(self, df: pd.DataFrame) -> "CleanPipeline":
        X = df.copy()
        self.fitted_steps.clear()
        for step in self.steps:
            name = step.__class__.__name__
            self.log.info("Fitting step: %s", name)
            if hasattr(step, "fit"):
                step.fit(X)
            X = step.transform(X)
            self.fitted_steps.append((name, step))
        return self

    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        X = df.copy()
        for name, step in self.fitted_steps:
            self.log.info("Applying step: %s", name)
            X = step.transform(X)
        return X

    def fit_transform(self, df: pd.DataFrame) -> pd.DataFrame:
        return self.fit(df).transform(df)

    def save(self, path: str):
        with open(path, "wb") as f:
            pickle.dump(self.fitted_steps, f)

    def load(self, path: str):
        with open(path, "rb") as f:
            self.fitted_steps = pickle.load(f)
        return self
