from hornero_event_classifier.pipelines import classify
from hornero_event_classifier import tools
from hornero_event_classifier.config import CONFIG
from utils import load_default_classifier, open_vid
from os import listdir, remove
import time
import pandas as pd
import numpy as np


RESTART_CLASSIFICATION: bool = True

files = listdir(CONFIG.yolo_path)
t0 = time.time()
out: list[pd.DataFrame] = []

file_exists = CONFIG.results_path.exists()
if RESTART_CLASSIFICATION and file_exists:
    remove(CONFIG.results_path)
elif file_exists:
    old_data = pd.read_csv(CONFIG.results_path)
    out.append(old_data)
    already_precessed = np.unique(old_data["video_id"])
    files = [file for file in files if tools.get_video_id(CONFIG.yolo_path / file) not in already_precessed]

for file in files:
    results, scene = classify(CONFIG.yolo_path / file, load_default_classifier())
    results.to_csv(CONFIG.results_path, index=False, header=not CONFIG.results_path.exists(), mode="a")
    out.append(results)



results: pd.DataFrame = pd.concat(out)
results = pd.read_csv(CONFIG.results_path)
print(f"total time: {time.time()-t0}s")
plot = tools.EventPlot(results)
plot.set_open_func(open_vid)
plot.show()
