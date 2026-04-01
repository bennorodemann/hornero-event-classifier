import pandas as pd

import hornero_event_classifier as hec
from pipelines import validate_events
from utils import open_vid
from paths import BORIS_FILE, RESULTS_FILE, METADATA_FILE

boris = pd.read_csv(BORIS_FILE)
df = pd.read_csv(RESULTS_FILE)
metadata_repo = hec.tools.read_metadata(METADATA_FILE)
results = validate_events(df, boris)
plot = hec.tools.EventPlot(metadata_repo, results)
plot.set_open_func(open_vid)
plot.show()
