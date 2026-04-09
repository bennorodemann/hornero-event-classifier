"""Utilities for estimating metric weights from validated event data.

Intended for internal calibration and model selection workflows.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import sklearn

from sklearn.linear_model import LogisticRegression, LogisticRegressionCV
from sklearn.pipeline import Pipeline, make_pipeline
from sklearn.preprocessing import StandardScaler


from hornero_event_classifier.classifiers import Metric

sklearn.set_config(enable_metadata_routing=True)


def classify_with_boris(yolo: pd.DataFrame, boris: pd.DataFrame) -> pd.DataFrame:
    """Compare segment data to BORIS annotations and estimate subject labels for each segment.

    This function attempts to match YOLO-derived segment rows with BORIS ground truth and label the resulting
    segments as ``ring`` or ``no_ring`` based on overlap and metric behavior.

    Output dataframe columns:
        - video_id: video name string
        - id: segment row id (used for grouping)
        - start_frame: first frame in segment
        - end_frame: last frame in segment
        - subject: BORIS subject label

    Input columns expected in ``yolo``:
        - video_id
        - start_frame
        - end_frame
        - one column per metric (uppercase names)

    Input columns expected in ``boris``:
        - video_id
        - subject
        - start_frame
        - end_frame

    :param yolo: segment data from :py:meth:`.SegmentCollection.as_df`
    :type yolo: pd.DataFrame
    :param boris: ground truth boris data
    :type boris: pd.DataFrame
    :return: classified segments
    :rtype: pd.DataFrame
    :seealso: :py:func:`recommend_weights`
    """
    yolo = yolo.copy()
    # remove non hornero events
    boris = boris[boris["subject"] != "otra_ave"]
    # add a row id (for later grouping)
    yolo.insert(1, "id", range(len(yolo)))
    # inner join yolo and boris dataframes by video_id
    df = pd.merge(yolo, boris, on="video_id", suffixes=("_yolo", "_boris"))
    df["length"] = df["end_frame_yolo"] - df["start_frame_yolo"] + 1
    # average of all metric scores
    df["score"] = df[[col for col in df.columns if col.isupper()]].mean(axis=1)
    # get percent overlap between yolo segment and boris events
    df["overlap"] = (
        df[["end_frame_yolo", "end_frame_boris"]].min(axis=1) - df[["start_frame_yolo", "start_frame_boris"]].max(axis=1) + 1
    ) / df["length"]
    # keep rows with segments longer than 100 frames and 70% overlap with a boris event
    df = df[(df["length"] > 100) & (df["overlap"] > 0.7)]
    # how many events did each segment overlap with
    df["n_result"] = df.groupby("id").transform("size")
    # keep rows with segments that only overlapped with one boris event or select the boris event that best matches with
    # PER_OWNERSHIP metric
    df = df[
        (df["n_result"] == 1)
        | ((df["subject"] == "ring") & (df["PER_OWNERSHIP"] >= 0.5))
        | ((df["subject"] == "no_ring") & (df["PER_OWNERSHIP"] < 0.5))
    ]
    # clean dataframe
    df = df.loc[:, "video_id":"subject"]
    df = df.rename(columns={"start_frame_yolo": "start_frame", "end_frame_yolo": "end_frame"})
    return df


def _get_weights(
    model: LogisticRegression | Pipeline, data: pd.DataFrame, metrics: list[str]
) -> tuple[np.float64, pd.Series[np.float64]]:
    """Fit a logistic model and compute normalized metric weights.

    :param model: Logistic regression model or pipeline.
    :type model: LogisticRegression | Pipeline
    :param data: Reference dataframe containing metric columns and ``subject`` labels.
    :type data: pd.DataFrame
    :param metrics: Metric column names to use.
    :type metrics: list[str]
    :return: ``(threshold, weights)`` where ``weights`` is a normalized series indexed by metric name.
    :rtype: tuple[np.float64, pd.Series[np.float64]]
    """
    # pull selected metrics
    X: pd.DataFrame = data[metrics]
    # set predictions
    y: pd.Series = (data["subject"] == "ring").astype(np.int64)
    # fit statistical model
    model.fit(X, y)
    # get model object
    if isinstance(model, Pipeline):
        model = model.named_steps[next(k for k in model.named_steps.keys() if k.startswith("logisticregression"))]
    # get coefficients
    weights: pd.Series[np.float64] = pd.Series(model.coef_[0], index=X.columns)  # type: ignore
    # get weight scaler
    weights_sum: np.float64 = weights.sum()
    # rescale weights
    weights = weights / weights_sum
    # get intercept and rescale
    threshold: np.float64 = -model.intercept_[0] / weights_sum  # type: ignore
    return threshold, weights


def recommend_weights(ref: pd.DataFrame, metrics: list[Metric] | None = None) -> tuple[np.float64, pd.Series[np.float64]]:
    """Apply a glm to output of :py:func:`classify_with_boris` to get recommended weights for selected metrics.

    :param ref: reference data
    :type ref: pd.DataFrame
    :param metrics: metrics to get weights for, if ``None`` (the default) all metrics in :py:class:`.Metric` are used
    :type metrics: list[Metric] | None, optional
    :return: a recommended threshold and weights for provided metrics
    :rtype: tuple[np.float64, pd.Series[np.float64]]
    :seealso: :py:func:`classify_with_boris`, :py:func:`~hornero_event_classifier.tools.validate_events.validate_events`
    """
    if not metrics:
        # get all metrics
        metric_strs: list[str] = [metric.name for metric in Metric]
        # fit a sparse logistic model to select non-zero metric weights
        _, weights = _get_weights(
            make_pipeline(
                StandardScaler(),  # scale input data
                LogisticRegressionCV(  # glmnet LASSO
                    solver="saga",  # required for l1_ratio
                    l1_ratios=[1.0],  # 1.0 = LASSO
                    Cs=20,  # tries 20 values of C
                    cv=10,  # 10-fold cross-validation
                    max_iter=5000,
                    n_jobs=-1,
                ),
            ),
            ref,
            metric_strs,
        )
        # filter for metrics that have a weight not equal to 0
        metric_strs = list(weights[weights != 0].index)
    else:
        # turn provided metrics into strings
        metric_strs = [metric.name for metric in metrics]
    # get weights and threshold
    return _get_weights(
        LogisticRegression(solver="lbfgs", C=np.inf, max_iter=1000),  # glm model
        ref,
        metric_strs,
    )
