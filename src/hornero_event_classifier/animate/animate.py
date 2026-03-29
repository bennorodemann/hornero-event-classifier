from __future__ import annotations

import time
from threading import Event, Thread, Timer
from typing import Callable, Optional, SupportsInt, TYPE_CHECKING, Literal
from pathlib import Path
import cv2
import hornero_event_classifier.classifiers.pre_calc as ref
import numpy as np
from hornero_event_classifier.animate.utils import ComplexEvent
from hornero_event_classifier.core.data import BBox, Frame, ItemType
from hornero_event_classifier.core.utils import FrameIndexer
from hornero_event_classifier.tools import get_video_path
from numpy.typing import NDArray

if TYPE_CHECKING:
    from hornero_event_classifier.core.scene import Scene

type Color = tuple[int, int, int]


class FramePos:
    def __get__(self, obj: Renderer, _=None) -> int:
        return obj._pos

    def __set__(self, instance: Renderer, value_: SupportsInt):
        value: int = int(value_)
        if instance.out_video:
            value = min(value, instance.written_frames + 1)
        value = max(value, instance.min_pos, -1)
        value = min(value, instance.max_pos)

        if value != instance._pos and instance._frame_ready.is_set():
            instance._pos = value
            instance._frame_ready.clear()


class AutoRefresher[T]:
    def __set_name__(self, owner: Renderer, name):
        # pylint: disable=[attribute-defined-outside-init]
        self.public_name = name
        self.private_name = "_" + name

    def __get__(self, instance: Renderer, _=None) -> T:
        return getattr(instance, self.private_name)

    def __set__(self, instance: Renderer, value: T):
        setattr(instance, self.private_name, value)
        if instance.paused:
            instance.refresh_frame()


class InputController:
    def __init__(self) -> None:
        self._event = Event()

    def wait(self, timeout: int) -> bool:
        return self._event.wait(timeout)


class Renderer:
    pos: FramePos = FramePos()

    def __init__(self, in_video: Path, out_video: Optional[str], box_data: FrameIndexer[Frame], scaler: float = 1.0):
        self.in_video = cv2.VideoCapture(in_video)
        self.video_length = self.in_video.get(cv2.CAP_PROP_FRAME_COUNT) - 1
        self._max_pos = int(self.video_length)
        self._min_pos = 0
        self._pos: int = -5
        if scaler <= 0:
            raise ValueError("scaler must be greater than 0")
        self._scaler: float = scaler
        w = int(self.in_video.get(cv2.CAP_PROP_FRAME_WIDTH) * scaler)
        h = int(self.in_video.get(cv2.CAP_PROP_FRAME_HEIGHT) * scaler)
        self._rescale: Callable[[NDArray], NDArray]
        if scaler == 1:
            self._rescale = lambda f: f
        else:
            self._rescale = lambda f: cv2.resize(f, (w, h))
        if not self.in_video.isOpened():
            raise ValueError(f"{in_video} could not be found")
        self.out_video: cv2.VideoWriter | None = None
        if out_video:
            fps = self.in_video.get(cv2.CAP_PROP_FPS)
            self.out_video = cv2.VideoWriter(out_video, cv2.VideoWriter.fourcc(*"mp4v"), fps, (w, h))
        self.box_data = box_data

        self.current_frame: NDArray = np.zeros((h, w), np.uint16)

        self._show_ignored: bool = True
        self._show_birds: bool = True
        self._show_rings: bool = True
        self._show_events: bool = True

        self.open: bool = True
        self.paused: bool = True
        self._show_boxes: bool = True
        self._frame_ready: ComplexEvent = ComplexEvent()
        self.written_frames: int = -1
        self.pos = 0
        self.render_frame()

        self.thread = Thread(target=self.render_loop, daemon=True)
        self.thread.start()

    @property
    def scaler(self) -> float:
        return self._scaler

    @property
    def rescale(self) -> Callable[[NDArray], NDArray]:
        return self._rescale

    show_boxes = AutoRefresher[bool]()
    show_ignored = AutoRefresher[bool]()
    show_birds = AutoRefresher[bool]()
    show_rings = AutoRefresher[bool]()
    show_events = AutoRefresher[bool]()

    @property
    def max_pos(self) -> int:
        return self._max_pos

    @max_pos.setter
    def max_pos(self, val: SupportsInt | None):
        if val is None or int(val) > self.video_length:
            self._max_pos = int(self.video_length)
        else:
            self._max_pos = int(val)
        if self._max_pos < self.pos:
            self.jump_to(self._max_pos)

    @max_pos.deleter
    def max_pos(self):
        self._max_pos = int(self.video_length)

    @property
    def min_pos(self):
        return self._min_pos

    @min_pos.setter
    def min_pos(self, val: SupportsInt | None):
        if val is None or int(val) < 0:
            self._min_pos = 0
        else:
            self._min_pos = int(val)
        if self._min_pos > self.pos:
            self.jump_to(self._min_pos)

    @min_pos.deleter
    def min_pos(self):
        self._min_pos = 0

    def jump_to(self, frame: int):
        if frame > self.max_pos:
            self._pos = self.max_pos
            self.written_frames = self._pos
        elif frame < self.min_pos:
            self._pos = self.min_pos
            self.written_frames = self._pos
        else:
            self._pos = frame
        self.written_frames = self._pos - 1
        while not self.frame_ready:
            time.sleep(0.1)
        self._frame_ready.clear()

    def jump_to_start(self):
        self.jump_to(self.min_pos)

    def jump_to_end(self):
        self.jump_to(self.max_pos)

    @property
    def frame_ready(self):
        return self._frame_ready.is_set()

    def grab_frame(self):
        next_frame = self.in_video.get(cv2.CAP_PROP_POS_FRAMES)
        jump = next_frame - self.pos
        if jump == 0:
            suc, frame = self.in_video.read()
            if not suc:
                self.pos -= 1
        else:
            self.in_video.set(cv2.CAP_PROP_POS_FRAMES, self.pos)
            suc, frame = self.in_video.read()
            self.pos = self.in_video.get(cv2.CAP_PROP_POS_FRAMES) - 1
            if not suc:
                self.pos -= 1
        return suc, frame

    def render_frame(self):
        self._frame_ready.wait_for_clear()
        success, frame = self.grab_frame()
        if success:
            frame = self._rescale(frame)
            if self.show_boxes:
                target = int(self.pos + 1)
                if self.box_data.has(target):
                    self._animate_frame(self.box_data[target], frame)
            self.write_frame(frame)
            self.current_frame = frame
        self._frame_ready.set()

    def _animate_frame(self, frame: Frame, img: NDArray):
        if self.show_ignored:
            for old in frame.orphans:
                self._animate_bbox(old, img, (0, 0, 255), show_id=False)
        if self.show_birds:
            for bird in frame.birds:
                self._animate_bbox(bird, img, (0, 255, 0 if bird.real else 255), show_center=True)
                if self.show_rings:
                    for ring in bird.metrics_cache.get(ref.local_rings, []):
                        cv2.line(
                            img,
                            (int(bird.x * self._scaler), int(bird.y * self._scaler)),
                            (int(ring.x * self._scaler), int(ring.y * self._scaler)),
                            (0, 255, 0),
                            thickness=int(self._scaler * 2),
                        )
        if self.show_rings:
            for ring in frame.rings:
                color = (255, 0, 0) if ring.item_obj.type == ItemType.RING_PLASTIC else (150, 150, 150)
                self._animate_bbox(ring, img, color, show_id=False)
        if self.show_events:
            for event in frame.events:
                text = event.item_obj.subject.value
                text = f"{event.item_obj.id}.{event.item_obj.sub_id}: {text}"
                self._animate_bbox(event, img, (0, 0, 0), (255, 255, 255), "nw", show_id=True, text_override=text)

    def _animate_bbox(
        self,
        bbox: BBox,
        img: NDArray,
        color: Color,
        text_color: Color = (0, 0, 0),
        text_anchor: Literal["ne", "nw", "se", "sw"] = "ne",
        text_override: str | None = None,
        show_center: bool = False,
        show_id: bool = True,
        alpha: float = 1.0,
        buffer: int = 0,
    ):
        if alpha < 1.0:
            original_img = img
            img = original_img.copy()
        x = int(bbox.x * self._scaler)
        y = int(bbox.y * self._scaler)
        xmin = int(bbox.xmin * self._scaler)
        xmax = int(bbox.xmax * self._scaler)
        ymin = int(bbox.ymin * self._scaler)
        ymax = int(bbox.ymax * self._scaler)
        buffer = int(buffer * self._scaler)
        if show_center:
            cv2.circle(img, (x, y), int(10 * self._scaler), color, -1)
        cv2.rectangle(img, (xmin - buffer, ymin - buffer), (xmax + buffer, ymax + buffer), color, int(5 * self._scaler))
        if show_id:
            text = text_override or f"{bbox.item_obj.id}.{bbox.item_obj.sub_id}({bbox.conf:.02f})"
            (text_width, text_height), text_base = cv2.getTextSize(
                text, cv2.FONT_HERSHEY_SIMPLEX, int(1 * self._scaler), int(2 * self._scaler)
            )
            text_height += 20
            text_base *= 2
            text_base += 10
            match text_anchor:
                case "ne":
                    x, y = (xmin - buffer, ymin - buffer)
                    rect_pt2 = (x + text_width, y + text_height)
                    text_pos = (x, y + text_base)
                case "nw":
                    x, y = (xmax + buffer, ymin - buffer)
                    rect_pt2 = (x - text_width, y + text_height)
                    text_pos = (x - text_width, y + text_base)
                case "se":
                    x, y = (xmin - buffer, ymax + buffer)
                    rect_pt2 = (x + text_width, y - text_height)
                    text_pos = (x, y - text_height + text_base)
                case "sw":
                    x, y = (xmax + buffer, ymax + buffer)
                    rect_pt2 = (x - text_width, y - text_height)
                    text_pos = (x - text_width, y - text_height + text_base)
                case _:
                    raise ValueError("Unsupported text anchor")
            cv2.rectangle(img, (x, y), rect_pt2, color, -1)
            cv2.putText(
                img,
                text,
                text_pos,
                cv2.FONT_HERSHEY_SIMPLEX,
                int(1 * self._scaler),
                text_color,
                int(2 * self._scaler),
                cv2.LINE_AA,
            )
        if alpha < 1.0:
            cv2.addWeighted(img, alpha, original_img, 1 - alpha, 0, dst=original_img)

    def refresh_frame(self):
        if not self.frame_ready:
            self._frame_ready.wait_for_set()
        self._frame_ready.clear()

    def write_frame(self, frame: NDArray):
        if self.out_video:
            self.out_video.write(frame)
        self.written_frames += 1

    def render_loop(self):
        while self.open:
            self.render_frame()

    def close(self):
        self.in_video.release()
        if self.out_video:
            self.out_video.release()
        self._frame_ready.set()
        self.open = False
        self._frame_ready.clear()
        self.thread.join()


class Animator:
    NORMAL: int = 0
    FRAME_JUMP: int = 1

    def __init__(
        self,
        scene: Scene,
        out_video: Optional[str] = None,
        mask: Optional[NDArray] = None,
        scale: float = 1.0,
    ):
        self.scene = scene
        self.open: bool = True
        self.mask = mask
        self.renderer = Renderer(get_video_path(scene.video_id), out_video, scene.frames, scaler=scale)
        self.rendered_frame = None
        self.min_sleep_time: int = 1  # 33
        self.last_render_time: float = 0
        self.paused: bool = False
        self.state: int = self.NORMAL
        self.text_entry: str = ""
        self._start: Optional[int] = None
        self._end: Optional[int] = None
        self.clipped = False
        self.layers_str: str = ""
        self._refresh_layers_str()

        cv2.namedWindow("out", cv2.WINDOW_NORMAL)
        cv2.imshow("out", self.renderer.current_frame)
        cv2.resizeWindow("out", 1920, 1080)
        self.update_window_name()

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, tb):
        self.renderer.close()
        self.close()

    @property
    def clipped(self) -> bool:
        return self._clipped

    @clipped.setter
    def clipped(self, val: bool):
        self._clipped = val
        if self._clipped:
            self.renderer.min_pos = self._start
            self.renderer.max_pos = self._end
        else:
            self.renderer.min_pos = None
            self.renderer.max_pos = None

    def _refresh_layers_str(self):
        self.layers_str = ""
        if self.renderer.show_boxes:
            self.layers_str += "layers: "
            self.layers_str += "I" if self.renderer.show_ignored else "_"
            self.layers_str += "B" if self.renderer.show_birds else "_"
            self.layers_str += "R" if self.renderer.show_rings else "_"
            self.layers_str += "E" if self.renderer.show_events else "_"

    def set_frame(self, val: int) -> None:
        self.renderer.jump_to(val)

    def set_start(self, val: Optional[int] = None):
        self._start = val
        if self._clipped:
            self.renderer.min_pos = self._start

    def set_end(self, val: Optional[int] = None):
        self._end = val
        if self._clipped:
            self.renderer.max_pos = self._end

    def display_frames(self):
        while self.open and self.renderer.open:
            if self.renderer.frame_ready and (
                (time.time() - self.last_render_time) > (self.min_sleep_time / 1000) or self.paused
            ):
                if self.renderer.current_frame is not self.rendered_frame:
                    c_frame: NDArray = self.renderer.current_frame
                    cv2.imshow("out", c_frame)
                if not self.paused:
                    if self.renderer.pos >= self.renderer.max_pos:
                        self.paused = True
                    else:
                        self.renderer.pos += 1
                        self.last_render_time = time.time()
                wait_time = 10 if self.paused else self.min_sleep_time
                self.update_window_name()
            else:
                wait_time = 1
            key = cv2.waitKey(wait_time)
            if self.state == self.NORMAL:
                self._normal_key_input(key)
            else:
                self._frame_jump_key_input(key)
            # detect if user closed the window
            try:
                if cv2.getWindowProperty("out", cv2.WND_PROP_VISIBLE) < 1:
                    self.close()
            except cv2.error:
                self.close()

    def _normal_key_input(self, key: int):
        match key:
            case -1:
                pass
            case 27:  # ESCAPE
                self.close()
            case 101:  # E
                if self.paused and self.renderer.frame_ready:
                    self.renderer.pos += 30
                    self.update_window_name()
            case 69:  # SHIFT + E
                if self.paused and self.renderer.frame_ready:
                    self.renderer.pos += 60
                    self.update_window_name()
            case 119:  # W
                self.min_sleep_time += 1
                self.update_window_name()
            case 113:  # Q
                if self.paused and self.renderer.frame_ready:
                    self.renderer.pos -= 30
                    self.update_window_name()
            case 81:  # SHIFT + Q
                if self.paused and self.renderer.frame_ready:
                    self.renderer.pos -= 60
                    self.update_window_name()
            case 100:  # D
                if self.paused and self.renderer.frame_ready:
                    self.renderer.pos += 1
                    self.update_window_name()
            case 68:  # SHIFT + D
                if self.paused and self.renderer.frame_ready:
                    self.renderer.jump_to_end()
            case 115:  # S
                if self.min_sleep_time > 1:
                    self.min_sleep_time -= 1
                    self.update_window_name()
            case 97:  # A
                if self.paused and self.renderer.frame_ready:
                    self.renderer.pos -= 1
            case 65:  # SHIFT + A
                if self.paused and self.renderer.frame_ready:
                    self.renderer.jump_to_start()
            case 32:  # SPACE
                self.paused = not self.paused
                self.update_window_name()
            case 106:  # J
                if self.paused:
                    self.text_entry = "0"
                    self.state = self.FRAME_JUMP
            case 99:  # C
                self.clipped = not self.clipped
            case 104:  # H
                self.renderer.show_boxes = not self.renderer.show_boxes
                self._refresh_layers_str()
            case 49 | 156:  # 1 | numpad 1
                self.renderer.show_ignored = not self.renderer.show_ignored
                self._refresh_layers_str()
            case 50 | 153:  # 2 | numpad 2
                self.renderer.show_birds = not self.renderer.show_birds
                self._refresh_layers_str()
            case 51 | 155:  # 3 | numpad 3
                self.renderer.show_rings = not self.renderer.show_rings
                self._refresh_layers_str()
            case 52 | 150:  # 4 | numpad 4
                self.renderer.show_events = not self.renderer.show_events
                self._refresh_layers_str()
            case other:
                print(other)

    def _frame_jump_key_input(self, key: int):
        match key:
            case 13:  # ENTER
                if self.text_entry:
                    self.renderer.jump_to(int(self.text_entry))
                self.state = self.NORMAL
                self.text_entry = "0"
            case 8:  # BACKSPACE
                self.text_entry = self.text_entry[:-1]
                if not self.text_entry:
                    self.text_entry = "0"
            case 113:  # Q
                self.state = self.NORMAL
            case _:
                for k in range(10):
                    if key == ord(str(k)):
                        self.text_entry += str(k)
                if self.text_entry.startswith("0"):
                    self.text_entry = self.text_entry[1:]

    def update_window_name(self):
        if self.state == self.FRAME_JUMP:
            cv2.setWindowTitle(
                "out",
                f"{self.scene.video_id} (jump to: {self.text_entry}{", Clipped" if self.clipped else ""}) {self.layers_str}",
            )
        elif self.paused:
            cv2.setWindowTitle(
                "out",
                f"{self.scene.video_id} (frame: {self.renderer.pos}{", Clipped" if self.clipped else ""}) {self.layers_str}",
            )
        else:
            cv2.setWindowTitle(
                "out",
                f"{self.scene.video_id} (sleep: {self.min_sleep_time} ms{", Clipped" if self.clipped else ""}) {self.layers_str}",
            )

    def close(self):
        self.open = False
        self.renderer.close()
        try:
            cv2.destroyWindow("out")
        except cv2.error:
            pass
