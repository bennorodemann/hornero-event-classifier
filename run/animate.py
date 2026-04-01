import hornero_event_classifier as hec
from utils import load_default_classifier
from pipelines import classify, animate
from paths import METADATA_FILE

TARGET_VIDEO: str = "n10_d4_c1_1_cl2"
metadata_repo = hec.tools.read_metadata(METADATA_FILE)
_, scene = classify(metadata_repo[TARGET_VIDEO], load_default_classifier())
animate(scene)
