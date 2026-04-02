from paths import METADATA_FILE
from pipelines import animate, classify
from utils import load_default_classifier

import hornero_event_classifier as hec

TARGET_VIDEO: str = "n10_d4_c1_1_cl2"
metadata_repo = hec.read_metadata(METADATA_FILE)
_, scene = classify(metadata_repo[TARGET_VIDEO], load_default_classifier())
animate(scene)
