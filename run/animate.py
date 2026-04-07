from paths import METADATA_FILE
from classify import classify, load_default_classifier

from hornero_event_classifier import Scene, ItemType, read_metadata, VideoMetadata
from hornero_event_classifier.tools import Animator


def animate(
    scene: Scene,
    scale: float = 1,
    frame: int | None = None,
    clip: tuple[int, int] | None = None,
    auto_play: bool = True,
    out_video: str | None = None,
):
    if clip and frame and not clip[0] <= frame <= clip[1]:
        raise ValueError(f"frame ({frame}) needs to be between clip values {clip}")
    video_path = scene.video_data.video_path
    if not video_path.exists():
        print(f"Video file not found: {video_path}")
        return
    scene.fill_gaps(None, ItemType.EVENT)
    with Animator(scene, out_video, scale=scale) as animator:
        if clip:
            animator.set_start(clip[0])
            animator.set_end(clip[1])
            animator.clipped = True
        if frame:
            animator.set_frame(frame)
        animator.paused = not auto_play
        animator.display_frames()


def event_plot_open_vid(_, video_metadata: VideoMetadata, mouse_pos: tuple[float, float]):
    if not video_metadata.video_path.exists():
        print(f"Video {video_metadata.name} not found at: {video_metadata.video_path}")
        return
    _, scene = classify(video_metadata, load_default_classifier(), show_progress=False)
    animate(scene, scale=2, frame=int(mouse_pos[0]), auto_play=False)
    return


if __name__ == "__main__":
    TARGET_VIDEO: str = "n10_d4_c1_1_cl2"
    metadata_repo = read_metadata(METADATA_FILE)
    _, scene = classify(metadata_repo[TARGET_VIDEO], load_default_classifier())
    animate(scene, scale=2)
