import csv
import os

import ffmpeg


def extract_metadata(video: str, probe_data: dict):
    stream = probe_data["streams"][[s["codec_type"] for s in probe_data["streams"]].index("video")]
    return {
        "video": video,
        "fps": stream["avg_frame_rate"],
        "duration_s": stream["duration"],
        "duration_f": int(float(stream["duration"]) * eval(stream["avg_frame_rate"])),
        "width": stream["width"],
        "height": stream["height"],
    }


def gen_video_metadata():
    nests = os.listdir("/home/bennor/Videos/videos_BORIS")
    metadata = []
    for nest in nests:
        vids = os.listdir(f"/home/bennor/Videos/videos_BORIS/{nest}")
        for vid in vids:
            metadata.append(extract_metadata(vid[:-4], ffmpeg.probe(f"/home/bennor/Videos/videos_BORIS/{nest}/{vid}")))
    print(metadata)
    with open("databases/general/video_metadata.csv", "w", newline="") as f:
        writer = csv.DictWriter(f, metadata[0].keys())
        writer.writeheader()
        writer.writerows(metadata)


if __name__ == "__main__":
    gen_video_metadata()
