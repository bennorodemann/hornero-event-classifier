from pathlib import Path
import hornero_event_classifier as hec
from utils import load_default_classifier

source = hec.CONFIG.yolo_path/"n10_d4_c1_1_cl2_bbox.csv"
_, scene = hec.pipelines.classify(source, load_default_classifier())
hec.pipelines.animate(scene)
