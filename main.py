from hornero_event_classifier.pipelines import classify, validate_events
from pathlib import Path
from os import listdir
import time

source = Path("databases/YOLOexp2")
target = Path("databases/pYOLOv3")
files = listdir(source)
t0 = time.time()
for file in files:
    results, scene = classify(source / file)
    save_file = file.replace("bbox", "events")
    results.to_csv(target / save_file, index=False)
print(f"total time: {time.time()-t0}")
validate_events()
