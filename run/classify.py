from hornero_event_classifier.pipelines import classify
from hornero_event_classifier import tools
from hornero_event_classifier import config
from hornero_event_classifier.classifiers import ThresholdClassifier, Metric
from os import listdir, remove
import time
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np


RESTART_CLASSIFICATION: bool = True

files = listdir(config.YOLO_PATH)
t0 = time.time()
out: list[pd.DataFrame] = []

file_exists = config.RESULT_PATH.exists()
if RESTART_CLASSIFICATION and file_exists:
    remove(config.RESULT_PATH)
elif file_exists:
    old_data = pd.read_csv(config.RESULT_PATH)
    out.append(old_data)
    already_precessed = np.unique(old_data["video_id"])
    files = [file for file in files if tools.get_video_id(config.YOLO_PATH / file) not in already_precessed]

for file in files:
    classifier = ThresholdClassifier.from_dict(
        {
            Metric.AVG_PLASTIC: 0.0359196367482639,
            Metric.AVG_Y_SCORE: 0.364215184150255,
            Metric.RING_PRESENCE: 0.597396820220843,
            Metric.RAD_STD: 0.377765678705804,
            Metric.AVG_RING_CONF: -0.575646002005343,
            Metric.PER_OWNERSHIP: 0.200348682180178,
        },
        0.2898628,
    )
    results, scene = classify(config.YOLO_PATH / file, classifier)
    results.to_csv(config.RESULT_PATH, index=False, header=not config.RESULT_PATH.exists(), mode="a")
    out.append(results)

results: pd.DataFrame = pd.concat(out)
results = pd.read_csv(config.RESULT_PATH)
print(f"total time: {time.time()-t0}s")
tools.plot_events(results)
plt.show()
