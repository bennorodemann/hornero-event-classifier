from pathlib import Path
from hornero_event_classifier.pipelines import animate


source = Path("databases/YOLOexp2/n4_d3_c1_4_bbox.csv")
animate(source)
