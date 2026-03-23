from hornero_event_classifier.tools.validate import get_overlap, validate_events, plot_validations, event_validation_str
from hornero_event_classifier.pipelines import classify

from pathlib import Path
from os import listdir
import time
import pandas as pd
import matplotlib.pyplot as plt

# source = Path("databases/YOLOexp2")
# target = Path("databases/pYOLOv3")
boris = pd.read_csv("databases/general/DB_BORIS.csv")
# files = listdir(source)
# t0 = time.time()
# dfs = []
# for file in files:  # [:2]:
#     data, _ = classify(source / file)
#     dfs.append(data)

# df = pd.concat(dfs)
# df.to_csv("databases/general/validate.csv")
df = pd.read_csv("databases/general/validate.csv")
overlaps = get_overlap(df, boris)
overlaps = validate_events(overlaps, 0.7)
print(event_validation_str(overlaps, True))
# print(overlaps)
fig, ax = plot_validations(overlaps)
plt.show()
