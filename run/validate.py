from pathlib import Path
import pandas as pd

import hornero_event_classifier as hec


boris = pd.read_csv(hec.config.BORIS_PATH)
df = pd.read_csv(hec.config.RESULT_PATH)
hec.pipelines.validate_events(df, boris)
