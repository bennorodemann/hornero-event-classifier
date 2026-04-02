import pandas as pd
from paths import BORIS_FILE, METADATA_FILE, RESULTS_FILE
from pipelines import validate_events
from utils import open_vid

import hornero_event_classifier as hec

boris = pd.read_csv(BORIS_FILE)
df = pd.read_csv(RESULTS_FILE)
metadata_repo = hec.read_metadata(METADATA_FILE)
results = validate_events(df, boris)
plot = hec.tools.EventPlot(metadata_repo, results)
plot.set_open_func(open_vid)
plot.show()
