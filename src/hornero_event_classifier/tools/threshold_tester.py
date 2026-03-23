import csv
import subprocess
import tkinter as tk
from os import listdir
from pathlib import Path
from typing import Any, Callable

import cv2
from hornero_event_classifier import ItemType, Scene, SegmentCollection, Metric, ThresholdClassifier
import hornero_event_classifier.core.filters as filters
from PIL import Image, ImageTk
from hornero_event_classifier.tools.eventval import run_all

db = Path("databases/YOLOexp2")
files = listdir(db)
metrics = tuple(Metric)
weights = [1 / len(metrics) for _ in metrics]
classifier = ThresholdClassifier(tuple(Metric), weights)
print("loading scenes...")
scenes = [
    Scene.from_csv(db / vid)
    .split_items(filters.make_gap_filter(100), ItemType.BIRD)
    .fill_gaps(filters.invert_filter(filters.frame_touch_filter), ItemType.BIRD)
    .split_items(filters.make_gap_filter(2), ItemType.BIRD)
    .split_items((filters.make_buffer_filter(100), filters.boundary_filter), ItemType.BIRD)
    .remove_low_conf(0.7, ItemType.BIRD)
    for vid in files
]
print("loading blocks...")
segment_data = [SegmentCollection(s.items.get(ItemType.BIRD), metrics) for s in scenes]
print("saving raw data...")
for scene, segments in zip(scenes, segment_data):
    segments.save_csv(scene.video_id)


def get_weight_setter(idx: int) -> Callable[[int], None]:
    def weight_setter(val: int) -> None:
        classifier.weights[idx] = val / 100

    return weight_setter


def set_threshold(val: int):
    classifier.threshold = val / 100


def request_refresh(*_):
    global refresh_requested
    refresh_requested = True


def refresh():
    print("grading...")
    new_weights = [v.get() for v in metric_vars]
    weights_sum = sum(new_weights)
    new_weights = [v / weights_sum for v in new_weights]
    new_thresh = thresh_var.get() / 100
    if weights_sum > 0 and (classifier.weights != new_weights or classifier.threshold != new_thresh):
        classifier.weights = new_weights
        classifier.threshold = new_thresh
        for scene, data in zip(scenes, segment_data):
            out = classifier.classify(data)

            with open(f"databases/pYOLOv3/{scene.video_id}_events.csv", "w", encoding="utf-8") as file:
                writer = csv.DictWriter(file, ("video_id", "subject", "start", "end", "mud"))
                writer.writeheader()
                for event in (block for item_blocks in out.values() for block in item_blocks):
                    # print(f"{event.key} ({'ringed' if event.grade else 'unringed'}): {event.start} -> {event.end}")
                    writer.writerow(
                        {
                            "video_id": scene.video_id,
                            "subject": "ring" if event.classification else "no_ring",
                            "start": event.start,
                            "end": event.end,
                            "mud": False,
                        }
                    )

    run_all(target_dir="pYOLOv3", overlap_threshold=overlap_var.get() / 100)

    result = subprocess.run(
        [
            "Rscript",
            "--vanilla",
            "analysis/R/event_validation_visual.R",
            "databases/pYOLOv3_validation",
            str(int(place_finder_var.get())),
        ],
        capture_output=True,
        text=True,
        check=True,
    )

    img = Image.open(result.stdout)
    w, h = img.size
    img = img.resize((int(w * 0.5), int(h * 0.5)))
    tk_img = ImageTk.PhotoImage(img)
    img_widget.config(image=tk_img)
    img_widget.image = tk_img  # type: ignore
    # return result.stdout
    # plot = cv2.imread(result.stdout)
    # print("showing")
    # if plot is not None:
    #     cv2.imshow("plot", plot)
    # else:
    #     raise FileNotFoundError(f"file not found: {result.stdout}")
    # img = Image.open(result.stdout)
    # fig, ax = plt.subplots(figsize=(19, 10))
    # ax.imshow(img)
    # ax.axis("off")
    # plt.show()


window = tk.Tk()
metric_vars = [tk.DoubleVar(None, v * 100) for v in weights]
thresh_var = tk.DoubleVar(None, 50)
overlap_var = tk.DoubleVar(None, 80)
place_finder_var = tk.DoubleVar(None, 0)
img_widget = tk.Label(window)
img_widget.pack(fill=tk.BOTH, side=tk.LEFT, expand=True)
controls = tk.Frame(window, bd=3)
control_widgets = []
for row, (metric, metric_var) in enumerate(zip(metrics, metric_vars)):
    if metric.name is None:
        continue
    label = tk.Label(controls, text=metric.name)
    slider = tk.Scale(controls, variable=metric_var, orient=tk.HORIZONTAL, to=10)
    label.grid(row=row, column=0)
    slider.grid(row=row, column=1)
    control_widgets.append(label)
    control_widgets.append(slider)
row += 1
label = tk.Label(controls, text="threshold")
slider = tk.Scale(controls, variable=thresh_var, orient=tk.HORIZONTAL)
label.grid(row=row, column=0)
slider.grid(row=row, column=1)
control_widgets.append(label)
control_widgets.append(slider)
row += 1
label = tk.Label(controls, text="overlap threshold")
slider = tk.Scale(controls, variable=overlap_var, orient=tk.HORIZONTAL)
label.grid(row=row, column=0)
slider.grid(row=row, column=1)
control_widgets.append(label)
control_widgets.append(slider)
row += 1
label = tk.Label(controls, text="place finder")
slider = tk.Scale(controls, variable=place_finder_var, orient=tk.HORIZONTAL, to=108_000)
label.grid(row=row, column=0)
slider.grid(row=row, column=1)
control_widgets.append(label)
control_widgets.append(slider)
row += 1
apply_button = tk.Button(controls, text="Apply", command=refresh)
apply_button.grid(row=row, column=0, columnspan=2)
controls.pack(side=tk.RIGHT, expand=True)  # .grid(row=0, column=1)
refresh()
window.mainloop()
# cv2.createTrackbar("threshold", "controls", int(grader.threshold * 100), 100, set_threshold)
# cv2.createButton("apply", request_refresh, None, cv2.QT_PUSH_BUTTON)


# while True:
#     if refresh_requested:
#         path = refresh()
#         plot = cv2.imread(path)
#         if plot is None:
#             raise FileNotFoundError(f"file not found: {path}")
#         plot = cv2.resize(plot, None, fx=0.4, fy=0.4)
#         cv2.imshow("plot", plot)
#         refresh_requested = False
#     key = cv2.waitKey(10) & 0xFF

#     if key == 27:
#         break
#     if key != 0:
#         print(key)

# cv2.waitKey(0)
