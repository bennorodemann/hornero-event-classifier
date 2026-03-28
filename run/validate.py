from pathlib import Path
import pandas as pd

import hornero_event_classifier as hec
from utils import open_vid

boris = pd.read_csv(hec.CONFIG.boris_path)
df = pd.read_csv(hec.CONFIG.results_path)
results = hec.pipelines.validate_events(df, boris)
plot = hec.tools.EventPlot(results)
plot.set_open_func(open_vid)
plot.show()