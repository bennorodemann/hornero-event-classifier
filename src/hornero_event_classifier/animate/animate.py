from __future__ import annotations

import math
import time
from threading import Event, Thread
from typing import Callable, Optional, SupportsInt
from pathlib import Path
import cv2
import hornero_event_classifier.classifiers.pre_calc as ref
import numpy as np
from hornero_event_classifier.animate.utils import ComplexEvent
from hornero_event_classifier.core.data import BBox, Frame, Item, ItemType
from hornero_event_classifier.core.scene import Scene
from hornero_event_classifier.core.utils import FrameIndexer
from numpy.typing import NDArray


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


class InputController:
    def __init__(self) -> None:
        self._event = Event()

    def wait(self, timeout: int) -> bool:
        return self._event.wait(timeout)


# class FramePos:
#     def __get__(self, obj: Renderer, type=None) -> float:
#         return obj._pos

#     def __set__(self, instance: Renderer, value: float):
#         if value > instance.written_frames + 1 and instance.out_video:
#             value = instance.written_frames + 1
#         if value < -1:
#             value = -1
#         if value > instance.max_pos:
#             value = instance.max_pos
#         if value < instance.min_pos:
#             value = instance.min_pos
#         if value != instance._pos and not instance._frame_ready.is_set():
#             instance._pos = value
#             instance._frame_ready.set()


class Renderer:
    pos: FramePos = FramePos()

    def __init__(self, in_video: str, out_video: Optional[str], box_data: FrameIndexer[Frame], scaler: float = 1.0):
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

        self.open: bool = True
        self.paused: bool = True
        self._show_boxes: bool = True
        self._frame_ready: ComplexEvent = ComplexEvent()
        self.written_frames: int = -1
        self.pos = 0
        self.global_point: tuple[int, int] | None = None
        self.render_frame()

        self.thread = Thread(target=self.render_loop, daemon=True)
        self.thread.start()

    @property
    def scaler(self) -> float:
        return self._scaler

    @property
    def rescale(self) -> Callable[[NDArray], NDArray]:
        return self._rescale

    @property
    def show_boxes(self):
        return self._show_boxes

    @show_boxes.setter
    def show_boxes(self, val: bool):
        self._show_boxes = val
        self.refresh_frame()

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
            if self.show_boxes:
                target = int(self.pos + 1)
                if self.box_data.has(target):
                    self._animate_frame(self.box_data[target], frame)
                    # self.box_data[self.pos + 1].animate(frame)
            frame = self.rescale(frame)
            self.write_frame(frame)
            self.current_frame = frame
        self._frame_ready.set()

    def _animate_frame(self, frame: Frame, img: NDArray):
        for old in frame.orphans:
            self._animate_bbox(old, img, (0, 0, 255), show_id=False)
        for bird in frame.birds:
            self._animate_bbox(bird, img, (0, 255, 0 if bird.real else 255), True)
            if self.global_point:
                cv2.line(img, (int(bird.x), int(bird.y)), self.global_point, (0, 255, 0))
                cv2.putText(
                    img,
                    str(int((math.atan2((self.global_point[0] - bird.x), -(self.global_point[1] - bird.y)) / math.pi) * 180)),
                    (int(bird.x), int(bird.y)),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    1,
                    (0, 0, 0),
                    2,
                    cv2.LINE_AA,
                )
            for ring in bird.metrics_cache[ref.local_rings]:
                cv2.line(img, (int(bird.x), int(bird.y)), (int(ring.x), int(ring.y)), (0, 255, 0))
                cv2.putText(
                    img,
                    str(int((math.atan2(ring.x - bird.x, -(ring.y - bird.y)) / math.pi) * 180)),
                    (int(ring.x), int(ring.y)),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    1,
                    (0, 0, 0),
                    2,
                    cv2.LINE_AA,
                )
            # cv2.line(img, (int(bird.x), int(bird.y)), (int(bird.xmin), int(bird.ymax)), (0, 255, 0))
            # cv2.line(img, (int(bird.x), int(bird.y)), (int(bird.xmax), int(bird.ymax)), (0, 255, 0))
            # angle = math.tan(math.pi / 4)
            # h = bird.ymax - bird.y
            # w = angle * h
            # cv2.line(img, (int(bird.x), int(bird.y)), (int(bird.x - w), int(bird.ymax)), (0, 0, 255))
            # cv2.line(img, (int(bird.x), int(bird.y)), (int(bird.x + w), int(bird.ymax)), (0, 0, 255))
        for ring in frame.rings:
            color = (255, 0, 0) if ring.item_obj.type == ItemType.RING_PLASTIC else (150, 150, 150)
            self._animate_bbox(ring, img, color, show_id=False)
        for event in frame.events:
            self._animate_bbox(event, img, (0, 0, 0), show_id=False)
            text = event.item_obj.subject.value
            text = f"{event.item_obj.id}.{event.item_obj.sub_id}: {text}"
            text_color = (255, 255, 255)
            (text_width, text_height), text_base = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 1, 2)
            text_height += 20
            text_base *= 2
            text_base += 10
            cv2.rectangle(
                img,
                (int(event.xmin), int(event.ymin - text_height)),
                (int(event.xmin + text_width), int(event.ymin)),
                (0, 0, 0),
                -1,
            )
            cv2.putText(
                img,
                text,
                (int(event.xmin), int(event.ymin - (text_base * 0.75))),
                cv2.FONT_HERSHEY_SIMPLEX,
                1,
                text_color,
                2,
                cv2.LINE_AA,
            )

    def _animate_bbox(
        self,
        bbox: BBox,
        img: NDArray,
        color: tuple[int, int, int],
        show_center: bool = False,
        show_id: bool = True,
    ):
        if show_center:
            cv2.circle(img, (int(bbox.x), int(bbox.y)), 10, color, -1)
        cv2.rectangle(img, (int(bbox.xmin), int(bbox.ymin)), (int(bbox.xmax), int(bbox.ymax)), color, 5)
        if show_id:
            text = f"{bbox.item_obj.id}.{bbox.item_obj.sub_id}({bbox.conf:.02f})"
            text_color = (0, 0, 0)
            (text_width, text_height), text_base = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 1, 2)
            text_height += 20
            text_base *= 2
            text_base += 10
            cv2.rectangle(
                img,
                (int(bbox.xmin), int(bbox.ymin)),
                (int(bbox.xmin + text_width), int(bbox.ymin + text_height)),
                color,
                -1,
            )
            cv2.putText(
                img,
                text,
                (int(bbox.xmin), int(bbox.ymin + text_base)),
                cv2.FONT_HERSHEY_SIMPLEX,
                1,
                text_color,
                2,
                cv2.LINE_AA,
            )

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

    def set_global_point(self, event, x, y, flags, param):
        if cv2.EVENT_FLAG_LBUTTON == flags:
            self.global_point = (x, y)

            self.refresh_frame()


class Animation:
    NORMAL: int = 0
    FRAME_JUMP: int = 1

    def __init__(
        self,
        scene: Scene,
        out_video: Optional[str] = None,
        mask: Optional[NDArray] = None,
        scale: float = 1.0,
        source: str | Path = Path.home() / "Videos/videos_BORIS",
    ):
        self.open: bool = True
        self.mask = mask
        self.renderer = Renderer(
            f"{source}/{scene.video_id.split("_",1)[0]}/{scene.video_id}.mp4", out_video, scene.frames, scaler=scale
        )
        self.rendered_frame = None
        self.min_sleep_time: int = 1  # 33
        self.last_render_time: float = 0
        self.paused: bool = False
        self.state: int = self.NORMAL
        self.text_entry: str = ""
        self._start: Optional[int] = None
        self._end: Optional[int] = None
        self.clipped = False
        cv2.namedWindow("out")
        cv2.setMouseCallback("out", self.renderer.set_global_point)
        cv2.imshow("out", self.renderer.current_frame)
        self.update_window_name()

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
                    # c_frame[self.mask] = (0, 0, 0)
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
            case other:
                print(other)

    def _frame_jump_key_input(self, key: int):
        match key:
            case 13:  # ENTER
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
            cv2.setWindowTitle("out", f" out (jump to: {self.text_entry}{", Clipped" if self.clipped else ""})")
        elif self.paused:
            cv2.setWindowTitle("out", f"out (frame: {self.renderer.pos}{", Clipped" if self.clipped else ""})")
        else:
            cv2.setWindowTitle("out", f"out (sleep: {self.min_sleep_time} ms{", Clipped" if self.clipped else ""})")

    def close(self):
        self.open = False
        self.renderer.close()
        cv2.destroyAllWindows()
