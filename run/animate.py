from pathlib import Path
from hornero_event_classifier.pipelines import animate


source = Path("databases/YOLOexp2/n10_d4_c1_1_cl2_bbox.csv")
animate(source)
