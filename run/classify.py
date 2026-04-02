import time
from os import remove

import numpy as np
import pandas as pd
from paths import RESULTS_FILE
from pipelines import classify
from utils import load_default_classifier, open_vid

from hornero_event_classifier import read_metadata, tools

RESTART_CLASSIFICATION: bool = True

t0 = time.time()
out: list[pd.DataFrame] = []

metadata_repo = read_metadata("data/video_metadata.json")

file_exists = RESULTS_FILE.exists()
if RESTART_CLASSIFICATION and file_exists:
    remove(RESULTS_FILE)
elif file_exists:
    old_data = pd.read_csv(RESULTS_FILE)
    out.append(old_data)
    already_precessed = np.unique(old_data["video_id"])
    video_ids = [video_id for video_id in metadata_repo if video_id not in already_precessed]

for file_metadata in metadata_repo.values():
    results, scene = classify(file_metadata, load_default_classifier())
    results.to_csv(RESULTS_FILE, index=False, header=not RESULTS_FILE.exists(), mode="a")
    out.append(results)


results: pd.DataFrame = pd.concat(out)
results = pd.read_csv(RESULTS_FILE)
print(f"total time: {time.time()-t0}s")
plot = tools.EventPlot(metadata_repo, results)
plot.set_open_func(open_vid)
plot.show()
