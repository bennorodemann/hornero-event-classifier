import pandas as pd
import numpy as np
from sklearn.linear_model import LogisticRegression, LogisticRegressionCV
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline, Pipeline
from hornero_event_classifier.classifiers import Metric
import sklearn
from numpy.typing import NDArray

sklearn.set_config(enable_metadata_routing=True)


def classify_with_boris(yolo: pd.DataFrame, boris: pd.DataFrame) -> pd.DataFrame:
    yolo = yolo.copy()
    boris = boris[boris["subject"] != "otra_ave"]
    yolo.insert(1, "id", range(len(yolo)))
    df = pd.merge(yolo, boris, on="video_id", suffixes=("_yolo", "_boris"))
    df["length"] = df["end_frame_yolo"] - df["start_frame_yolo"] + 1
    df["score"] = df[[col for col in df.columns if col.isupper()]].mean(axis=1)
    df["overlap"] = (
        df[["end_frame_yolo", "end_frame_boris"]].min(axis=1) - df[["start_frame_yolo", "start_frame_boris"]].max(axis=1) + 1
    ) / df["length"]
    df = df[(df["length"] > 100) & (df["overlap"] > 0.7)]
    df["n_result"] = df.groupby("id").transform("size")
    df = df[
        (df["n_result"] == 1)
        | ((df["subject"] == "ring") & (df["PER_OWNERSHIP"] >= 0.5))
        | ((df["subject"] == "no_ring") & (df["PER_OWNERSHIP"] < 0.5))
    ]
    df = df.loc[:, "video_id":"subject"]
    df = df.rename(columns={"start_frame_yolo": "start_frame", "end_frame_yolo": "end_frame"})
    return df


def _get_weights(
    model: LogisticRegression | Pipeline, data: pd.DataFrame, metrics: list[str]
) -> tuple[np.float64, pd.Series[np.float64]]:
    X: pd.DataFrame = data[metrics]
    y: pd.Series = (data["subject"] == "ring").astype(np.int64)
    model.fit(X, y)
    if isinstance(model, Pipeline):
        model = model.named_steps[next(k for k in model.named_steps.keys() if k.startswith("logisticregression"))]
    weights: pd.Series[np.float64] = pd.Series(model.coef_[0], index=X.columns)  # type: ignore
    weights_sum: np.float64 = weights.sum()
    weights = weights / weights_sum
    intercept: np.float64 = -model.intercept_[0] / weights_sum  # type: ignore
    return intercept, weights


def recommend_weights(ref: pd.DataFrame, metrics: list[Metric] | None = None) -> tuple[np.float64, pd.Series[np.float64]]:
    if not metrics:
        metric_strs: list[str] = [metric.name for metric in Metric]
        _, weights = _get_weights(
            make_pipeline(
                StandardScaler(),
                LogisticRegressionCV(
                    solver="saga",  # required for l1_ratio
                    l1_ratios=[1.0],  # 1.0 = LASSO
                    Cs=20,  # tries 20 values of C
                    cv=10,  # 5-fold cross-validation
                    max_iter=5000,
                    n_jobs=-1,
                ),
            ),
            ref,
            metric_strs,
        )
        metric_strs = list(weights[weights != 0].index)
    else:
        metric_strs = [metric.name for metric in metrics]
    intercept, weights = _get_weights(
        LogisticRegression(
            solver="lbfgs", C=np.inf, max_iter=1000  # stable, fast  # disables regularization (modern replacement)
        ),
        ref,
        metric_strs,
    )
    return intercept, weights
