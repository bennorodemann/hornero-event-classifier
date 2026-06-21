"""
Animation script for visualizing hornero event classification scenes.

This module provides functionality to animate video scenes with classified events,
allowing for playback, clipping, and frame-specific viewing.
"""

from argparse import ArgumentParser
from threading import Thread

import cv2
import numpy as np
from classify import classify, load_default_classifiers
from config import config

from hornero_event_classifier import ItemType, Scene, VideoMetadata, read_metadata
from hornero_event_classifier.tools import Animator


def animate(
    scene: Scene,
    scale: float = 1,
    frame: int | None = None,
    clip: tuple[int | None, int | None] | None = None,
    auto_play: bool = True,
    out_video: str | None = None,
):
    """
    Animate a video scene with classified events.

    This function creates an animation of the video scene, optionally scaling it,
    setting a specific starting frame, clipping to a time range, and controlling
    playback behavior.

    Args:
        scene: The Scene object containing video data and classified events.
        scale: Scaling factor for the animation display (default: 1).
        frame: Specific frame number to start playback from (default: None).
        clip: Tuple of (start_frame, end_frame) to clip the animation to (default: None).
        auto_play: Whether to start playback automatically (default: True).
        out_video: Path to save the animation as a video file (default: None for display only).

    Raises:
        ValueError: If frame is specified and not within the clip range.
    """
    # Validate frame is within clip bounds if both are provided
    if clip and frame and clip[0] is not None and clip[1] is not None and not clip[0] <= frame <= clip[1]:
        raise ValueError(f"frame ({frame}) needs to be between clip values {clip}")

    # Check if video file exists
    video_path = scene.video_data.file_path
    if not video_path.exists():
        print(f"Video file not found: {video_path}")
        return

    # Fill gaps in event data for smoother animation
    scene.fill_gaps(None, ItemType.EVENT)

    # Create animator context manager
    with Animator(scene, out_video, scale=scale) as animator:
        # Set clip boundaries if provided
        if clip:
            animator.set_start(clip[0])
            animator.set_end(clip[1])
            animator.clipped = True

        # Set starting frame if provided
        if frame:
            animator.set_frame(frame)

        # Control auto-play behavior
        animator.paused = not auto_play

        # Start the animation display
        animator.display_frames()


def event_plot_open_vid(_, video_metadata: VideoMetadata, mouse_pos: tuple[float, float]):
    """
    Callback function to open and animate a video at a specific frame from mouse position.

    This function is typically used as a click callback in event plots, allowing
    users to click on a timeline to view the corresponding video frame.

    Args:
        _: Unused parameter (typically event data).
        video_metadata: Metadata for the video to animate.
        mouse_pos: Tuple of (x, y) mouse coordinates, where x is used as frame number.
    """

    # create a extra window and hide it off screen
    # without this opencv's cleanup sometimes destroys matplotlib's windows
    cv2.namedWindow("DO NOT CLOSE")
    cv2.moveWindow("DO NOT CLOSE", -10000, -10000)

    # from multiprocessing import Process

    # Check if video file exists
    if not video_metadata.file_path.exists():
        print(f"Video {video_metadata.name} not found at: {video_metadata.file_path}")
        return

    # Classify the video scene without progress display
    classifiers = load_default_classifiers()
    _, scene = classify(
        video_metadata, classifiers["subject"], classifiers.get("mud", None), show_progress=False, remove_low_conf=0
    )

    # Animate starting from the clicked frame position, scaled up and paused
    Thread(target=animate, args=(scene,), kwargs={"scale": 2, "frame": int(mouse_pos[0]), "auto_play": False}).start()
    return


parser = ArgumentParser()
parser.add_argument("video", default="", help="video id, if incomplete finds closest match")
parser.add_argument("--scale", default=1, type=float, help="rescale ratio of original video")
parser.add_argument("--frame", type=int, help="starting frame")
parser.add_argument(
    "--clip",
    type=lambda v: [int(f) if f else None for f in v.split(",")],
    help="comma separated video frame boundaries",
)
parser.add_argument("--auto-play", action="store_true", help="automatically start playing when window opens")
parser.add_argument("--save", help="path to save output video to")

if __name__ == "__main__":
    # get input arguments
    args = parser.parse_args()
    # parse clip argument
    if args.clip is not None:
        if len(args.clip) == 1:
            args.clip.append(None)
        elif len(args.clip) > 2:
            raise ValueError("clip argument can not have more than 2 values")

    # Load video metadata repository
    metadata_repo = read_metadata(config.metadata_file)

    for metadata in metadata_repo:
        if metadata.startswith(args.video):
            target_video: str = metadata
            break
    else:
        raise ValueError(f"No video name could be found including: {args.video}")
    # Classify the target video and animate it
    classifiers = load_default_classifiers()
    _, scene = classify(metadata_repo[target_video], classifiers["subject"], classifiers.get("mud", None))
    animate(scene, scale=args.scale, frame=args.frame, clip=args.clip, auto_play=args.auto_play, out_video=args.save)
