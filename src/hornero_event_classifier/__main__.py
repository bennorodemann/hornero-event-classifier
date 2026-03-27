import subprocess
from cProfile import Profile
from os import listdir
from pathlib import Path
from pstats import SortKey, Stats
from time import time

import matplotlib.pyplot as plt
from PIL import Image
from validation_tools.eventval import run, run_all

from hornero_event_classifier import Animator, ItemType, Metric, Scene, ThresholdClassifier

db = Path("databases/YOLOexp2")
files = listdir(db)
t0 = time()
with Profile() as profile:
    for _ in range(30):
        for file in files:
            # if not file.startswith("n10_d4_c1_1_cl2"):
            #     continue
            # if not file.startswith("n5_d5"):
            #     continue
            print(f"{file}: ")
            t1 = time()
            s = Scene.from_csv(db / file)
            t2 = time()
            s.remove_low_conf(0.7)
            s.split_items(100)
            t3 = time()
            s.fill_gaps(True)
            s.split_items(2)
            t4 = time()
            # s.merge_birds(0.7, 0.7)
            t5 = time()
            # s.remove_minor_items(30)
            s.split_at_boundary(100)
            # s.split_at_frame_touch(100)
            t6 = time()
            s.classify(
                # KMeanGrader(
                #     Metric.LOCAL_RING_DISTRIBUTION
                #     | Metric.LOCAL_RING_CONF
                #     | Metric.LOCAL_RING_COUNT
                #     | Metric.LOCAL_PER_PLASTIC
                #     | Metric.AVG_LOCAL_X_SCORE
                #     | Metric.AVG_LOCAL_Y_SCORE
                # )
                ThresholdClassifier(
                    (
                        Metric.AVG_RAD_SCORE,
                        Metric.AVG_Y_SCORE,
                        Metric.RING_PRESENCE,
                        Metric.RAD_STD,
                        Metric.AVG_RING_CONF,
                        Metric.AVG_PLASTIC,
                    ),
                    # (843.1265, 2518.5255, 2376.7741, 1911.3064, -2836.3300),
                    # 1626.1397,
                    (1575.7270, 7020.6840, 3769.7068, 5581.0615, -1695.0269, 630.0441),
                    8087.7294,
                )
            )  # | Metric.RING_COUNT | Metric.CONF | Metric.REAL))
            # s.grade(
            #     grader := ThresholdGrader(
            #         Metric.LOCAL_RING_DISTRIBUTION
            #         | Metric.LOCAL_RING_COUNT
            #         | Metric.LOCAL_RING_CONF
            #         | Metric.LOCAL_RING_REAL
            #         | Metric.AVG_LOCAL_RING_POS_ANGLE
            #         | Metric.AVG_LOCAL_RING_POS_DISTANCE,
            #         300000,
            #         smooth=1,
            #     )
            # )
            t7 = time()
            s.define_events(90, 100)
            # s.remove_minor_items(100)
            t8 = time()
            s.write_to_csv()
            t9 = time()
            # KMeanGrader(Metric.GLOBAL_RING_DISTRIBUTION).save_blocks_csv(s.video_id, s.items.values())
            print(f"read: {t2 - t1}")
            print(f"split items: {t3 - t2}")
            print(f"fill gaps: {t4 - t3}")
            print(f"merge birds: {t5-t4}")
            print(f"remove minor: {t6 - t5}")
            print(f"grade: {t7 - t6}")
            print(f"define events: {t8 - t7}")
            print(f"write to csv: {t9 - t8}")
            print(f"total: {t9 - t1}")
            run(s.video_id, target_dir="pYOLOv3", overlap_threshold=0.9)
            print()

        # (Stats(profile).strip_dirs().sort_stats(SortKey.CALLS).print_stats())
        Stats(profile).dump_stats("profiles/eventer.prof")
print(f"timer: {time()-t0}")

# grader.evaluate()
# a = Animation(
#     s,
#     # "/home/bennor/Videos/missing_boris_bird_example.mp4",
#     source="/home/bennor/Videos/videos_BORIS",
#     scale=1,
# )
# # a.set_start(3909)
# # a.set_end(5467)
# # a.clipped = True
# a.display_frames()
# # run(s.video_id, target_dir="pYOLOv3")
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
