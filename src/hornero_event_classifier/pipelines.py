from hornero_event_classifier.core import Scene, ItemType
from hornero_event_classifier.animate.animate import Animation
from hornero_event_classifier.classifiers import ThresholdClassifier, Metric
import hornero_event_classifier.core.filters as filters
from pathlib import Path
import pandas as pd
from validation_tools.eventval import run_all
import subprocess
from PIL import Image
import matplotlib.pyplot as plt


def classify(file: str | Path) -> tuple[pd.DataFrame, Scene]:
    file = Path(file)
    filename: str = file.name
    print(f"{filename}: loading...         ", end="")
    s = Scene.from_csv(file)
    print(f"\r{filename}: pre-processing...", end="")
    (
        s.split_items(filters.make_gap_filter(100), ItemType.BIRD)
        .fill_gaps(filters.invert_filter(filters.frame_touch_filter), ItemType.BIRD)
        .split_items(filters.make_gap_filter(2), ItemType.BIRD)
        .split_items((filters.make_buffer_filter(100), filters.boundary_filter), ItemType.BIRD)
        .remove_low_conf(0.7, ItemType.BIRD)
    )
    print(f"\r{filename}: classifying...   ", end="")
    (
        s.classify(
            ThresholdClassifier(
                (
                    Metric.AVG_PLASTIC,
                    Metric.AVG_Y_SCORE,
                    Metric.RING_PRESENCE,
                    Metric.RAD_STD,
                    Metric.AVG_RING_CONF,
                    Metric.PER_OWNERSHIP,
                ),
                (
                    0.0359196367482639,
                    0.364215184150255,
                    0.597396820220843,
                    0.377765678705804,
                    -0.575646002005343,
                    0.200348682180178,
                ),
                0.2898628,
                # (
                #     Metric.AVG_RAD_SCORE,
                #     Metric.AVG_Y_SCORE,
                #     Metric.RING_PRESENCE,
                #     Metric.RAD_STD,
                #     Metric.AVG_RING_CONF,
                #     Metric.AVG_PLASTIC,
                # ),
                # (
                #     0.0490878247705511,
                #     0.467235453187786,
                #     0.383525333012307,
                #     0.463465766521598,
                #     -0.434597765850542,
                #     0.071283388358299,
                # ),
                # 0.3462933,
                # (1575.7270, 7020.6840, 3769.7068, 5581.0615, -1695.0269, 630.0441),
                # 8087.7294,
            )
        )
        .define_events(120, 100)
        .remove_minor_items(100, ItemType.EVENT)
    )
    print(f"\r{filename}: done             ")
    return s.get_results(), s


def animate(file: str | Path):
    _, s = classify(file)
    s.fill_gaps(None, ItemType.EVENT)
    a = Animation(s)
    a.display_frames()


def gen_video_metadata(): ...


def recommend_weights(): ...


def validate_events():
    run_all(target_dir="pYOLOv3", overlap_threshold=0.9)

    result = subprocess.run(
        ["Rscript", "--vanilla", "analysis/R/event_validation_visual.R", "databases/pYOLOv3_validation"],
        capture_output=True,
        text=True,
        check=True,
    )
    img = Image.open(result.stdout)
    fig, ax = plt.subplots(figsize=(19, 10))
    ax.imshow(img)
    ax.axis("off")
    plt.show()


def validate_frames(): ...


def validate_overlap(): ...
