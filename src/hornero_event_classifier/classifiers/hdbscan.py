"""Experimental HDBSCAN-based classifier prototype.

This module currently provides a placeholder :py:class:`HDBSCANClassifier` and retains exploratory analysis code in comments.
"""

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from hdbscan import HDBSCAN
from hornero_event_classifier.classifiers.kmean import KMeanClassifier
from sklearn.feature_selection import mutual_info_classif
from sklearn.preprocessing import StandardScaler


class HDBSCANClassifier(KMeanClassifier):
    """Placeholder HDBSCAN classifier.

    Currently inherits :py:class:`KMeanClassifier` behavior and serves as a stub for future development.
    """
    ...


# def _initiate(self):
#     self._data_scaler = StandardScaler()
#     self._model = HDBSCAN(min_cluster_size=6)

# def visualize(self):
#     if self.training_data is None:
#         raise ValueError("Grader must first be trained")
#     features: list[str] = [m.name for m in self.metrics if m.name]
#     df = pd.DataFrame(self._data_scaler.transform(self.training_data), columns=features)
#     X_scaled = self._data_scaler.transform(self.training_data)

#     clusterer = self._model

#     labels = clusterer.fit_predict(X_scaled)

#     df["cluster"] = labels
#     sns.pairplot(df, vars=features, hue="cluster", palette="tab10")

#     plt.show()

# def evaluate(self):
#     if self.training_data is None:
#         raise ValueError("Grader must first be trained")
#     X = self.training_data
#     features: list[str] = [m.name for m in self.metrics if m.name]
#     df = pd.DataFrame(X, columns=features)

#     X_scaled = self._data_scaler.transform(X)

#     clusterer = self._model

#     labels = clusterer.fit_predict(X_scaled)

#     df["cluster"] = labels

#     # remove noise for analysis
#     mask = labels != -1
#     X_valid = X_scaled[mask]
#     labels_valid = labels[mask]

#     # mutual information
#     mi = mutual_info_classif(X_valid, labels_valid)

#     results = pd.DataFrame({"feature": features, "mutual_information": mi}).sort_values("mutual_information", ascending=False)

#     print("\nFeature importance (Mutual Information):")
#     print(results)

#     # correlation matrix
#     corr = df[features].corr()

#     plt.figure(figsize=(8, 6))
#     sns.heatmap(corr, annot=True, cmap="coolwarm")
#     plt.title("Feature Correlation Matrix")
#     plt.show()

#     # cluster distributions
#     for f in features:

#         plt.figure()
#         sns.boxplot(x=df["cluster"], y=df[f])
#         plt.title(f"{f} distribution per HDBSCAN cluster")
#         plt.show()

#     return results
