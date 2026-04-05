from __future__ import annotations

import time
from pathlib import Path
from threading import Thread, Condition
from typing import TYPE_CHECKING, Callable, Literal, Optional, SupportsInt

import cv2
import numpy as np
from numpy.typing import NDArray

import hornero_event_classifier.classifiers.dependencies as ref
from hornero_event_classifier.core.collections import FrameIndexer
from hornero_event_classifier.core.data import BBox, Frame, ItemType

if TYPE_CHECKING:
    from hornero_event_classifier.core.scene import Scene

type Color = tuple[int, int, int]


class StateEvent:
    """This class is very similar to ``threading.Event``. The key differentiating feature is that other objects can wait for the
    ``StateEvent`` to be either set or cleared."""

    def __init__(self) -> None:
        self._flag = False
        self._cond = Condition()

    def set(self):
        """Set internal flag to ``True`` and notify all objects waiting at :py:meth:`StateEvent.wait_for_set`"""
        with self._cond:
            self._flag = True
            self._cond.notify_all()

    def clear(self):
        """Set internal flag to ``False`` and notify all objects waiting at :py:meth:`StateEvent.wait_for_clear`"""
        with self._cond:
            self._flag = False
            self._cond.notify_all()

    def is_set(self) -> bool:
        """Return the internal flags current state.

        :return: Internal flags value.
        :rtype: bool
        """
        return self._flag

    def wait_for_set(self, timeout: Optional[float] = None):
        """Wait for internal flag to be set to ``True``. If the internal flag is already ``True`` returns instantly.

        :param timeout: Time (in seconds) wait wait before giving up. If ``None`` (the default) it never timeout.
        :type timeout: Optional[float], optional
        """
        with self._cond:
            if self._flag is False:
                self._cond.wait(timeout)

    def wait_for_clear(self, timeout: Optional[float] = None):
        """Wait for internal flag to be set to ``False``. If the internal flag is already ``False`` returns instantly.

        :param timeout: Time (in seconds) wait wait before giving up. If ``None`` (the default) it never timeout.
        :type timeout: Optional[float], optional
        """
        with self._cond:
            if self._flag is True:
                self._cond.wait(timeout)


class AutoRefresher[T]:
    """A descriptor class that will automatically refresh the current frame when the attribute is set."""

    def __set_name__(self, owner: Renderer, name):
        # pylint: disable=[attribute-defined-outside-init]
        self.public_name = name
        self.private_name = "_" + name

    def __get__(self, instance: Renderer, _=None) -> T:
        return getattr(instance, self.private_name)

    def __set__(self, instance: Renderer, value: T):
        setattr(instance, self.private_name, value)
        if instance.paused:
            instance.trigger_render()


class Renderer:
    """This class handles the reading of an input video, rendering :py:class:`.Scene` info, and optionally writing the rendered
    frames to an output video.

    Renderings consist of :py:class:`~hornero_event_classifier.core.data.BBox` split up into 4 layers on top of the input videos
    frame that can be toggled off and on individually. They are (from bottom to top):

    1. :py:class:`.Item`\\s with the ignore flag in red (previous layers do not show ignored :py:class:`.Item`\\s).
    2. :py:attr:`~.ItemType.BIRD`\\s in green if the :py:class:`.BBox` is real or yellow if not real.
    3. :py:attr:`~.ItemType.RING_PLASTIC`\\s in blue and :py:attr:`~.ItemType.RING_METAL`\\s in gray.
    4. :py:attr:`~.ItemType.EVENT`\\s in black.

    If layers 2 and 3 are both active, and :py:attr:`~.ItemType.BIRD` has :py:func:`.local_rings` loaded, then lines are also
    drawn to connect ``bird`` and ``ring`` bounding :py:class:`.BBox`\\s.

    .. note::
        This class does not show the output of frames. For window to display rendered frames use :py:class:`Animator`.

    :param in_video: Path to source video file.
    :type in_video: Path | str
    :param frame_data: :py:class:`.BBox` info to render onto frames. Can be take from :py:class:`.Scene`.
    :type frame_data: FrameIndexer[Frame]
    :param out_video: Optional path to write output video to. If ``None`` (the default), no video is written.
    :type out_video: Optional[str]
    :param scaler: A rescale value to resize the frames with, defaults to 1.0.
    :type scaler: float, optional
    """

    def __init__(
        self,
        in_video: Path | str,
        frame_data: FrameIndexer[Frame],
        out_video: Optional[str] = None,
        scaler: float = 1.0,
    ):
        # initialize variables
        self.in_video = cv2.VideoCapture(in_video)
        self.video_length = self.in_video.get(cv2.CAP_PROP_FRAME_COUNT) - 1
        self._max_pos = int(self.video_length)
        self._min_pos = 0
        self._pos: int = -5
        # make sure not negative or 0 scaler
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
        # if out_video provided, initialize
        self._fps = int(self.in_video.get(cv2.CAP_PROP_FPS))
        if out_video:
            self.out_video = cv2.VideoWriter(out_video, cv2.VideoWriter.fourcc(*"mp4v"), self._fps, (w, h))
        self.frame_data = frame_data

        self.current_frame: NDArray = np.zeros((h, w), np.uint16)

        # initiate all layers
        self._show_ignored: bool = True
        self._show_birds: bool = True
        self._show_rings: bool = True
        self._show_events: bool = True

        self.open: bool = True
        self.paused: bool = True
        self._show_boxes: bool = True
        self._frame_ready: StateEvent = StateEvent()
        self.pos = 0
        self.render_frame()

        # start rendering while loop
        self.thread = Thread(target=self._render_loop, daemon=True)
        self.thread.start()

    @property
    def scaler(self) -> float:
        """The relative size of rendered frames compared to input video. Can not be set."""
        return self._scaler

    @property
    def rescale(self) -> Callable[[NDArray], NDArray]:
        """Function to rescale input video frames. Can not be set."""
        return self._rescale

    @property
    def fps(self) -> int:
        return self._fps

    @property
    def pos(self) -> int:
        """The current frame position. When setting, is the same as
        :py:meth:`Renderer.set_pos(value_, wait_til_ready=False) <Renderer.set_pos>`"""
        return self._pos

    @pos.setter
    def pos(self, value_: int):
        self.set_pos(value_, wait_til_ready=False)

    #: Weather to render :py:class:`BBox`\\s in frame. The current frame is automatically refreshed when this is set.
    show_boxes = AutoRefresher[bool]()
    #: Weather to render ignored layer (layer 1). The current frame is automatically refreshed when this is set.
    show_ignored = AutoRefresher[bool]()
    #: Weather to render birds layer (layer 2). The current frame is automatically refreshed when this is set.
    show_birds = AutoRefresher[bool]()
    #: Weather to render rings layer (layer 3). The current frame is automatically refreshed when this is set.
    show_rings = AutoRefresher[bool]()
    #: Weather to render events layer (layer 4). The current frame is automatically refreshed when this is set.
    show_events = AutoRefresher[bool]()

    @property
    def max_pos(self) -> int:
        """Maximum allowed frame position. When set, is clamped to the video length and must remain above
        :py:attr:`Renderer.min_pos`. If the current position exceeds the new maximum, the renderer jumps to it."""
        return self._max_pos

    @max_pos.setter
    def max_pos(self, val: SupportsInt | None):
        val = None if val is None else int(val)
        # if set value is none or more than video length, set max_pos to video length
        if val is None or val > self.video_length:
            self._max_pos = int(self.video_length)
        # set val if above min pos
        elif val > self.min_pos:
            self._max_pos = val
        else:
            raise ValueError("max_pos must be greater than min_pos.")
        # if current pos is beyond max_pos, set pos equal to max_pos
        if self._max_pos < self.pos:
            self.set_pos(self._max_pos)

    @max_pos.deleter
    def max_pos(self):
        self._max_pos = int(self.video_length)

    @property
    def min_pos(self):
        """Minimum allowed frame position. When set, is clamped to 0 and must remain above :py:attr:`Renderer.max_pos`. If the
        current position exceeds the new minimum, the renderer jumps to it."""
        return self._min_pos

    @min_pos.setter
    def min_pos(self, val: SupportsInt | None):
        val = None if val is None else int(val)
        # if set value is none or less than 0, set min_pos to 0
        if val is None or val < 0:
            self._min_pos = 0
        # if val is less than max pos
        elif val < self.max_pos:
            self._min_pos = val
        else:
            raise ValueError("min_pos must be less than max_pos.")
        # if current pos is less than min_pos, set pos equal to min_pos
        if self._min_pos > self.pos:
            self.set_pos(self._min_pos)

    @min_pos.deleter
    def min_pos(self):
        self._min_pos = 0

    @property
    def frame_is_ready(self) -> bool:
        """Return if the current frame is finished rendering."""
        return self._frame_ready.is_set()

    def set_pos(self, pos: int, wait_til_ready: bool = True):
        # make sure new value is within allowed boundaries.
        if pos > self.max_pos:
            pos = self.max_pos
        elif pos < self.min_pos:
            pos = self.min_pos
        # wait til frame is ready then render this frame
        if wait_til_ready:
            self._pos = pos
            self.trigger_render()
        # if there is a frame currently being rendered ignore input.
        elif self._frame_ready.is_set():
            self._pos = pos
            self._frame_ready.clear()

    def trigger_render(self):
        self._frame_ready.wait_for_set()
        self._frame_ready.clear()

    def jump_to_start(self):
        """Set :py:attr:`pos` equal to :py:attr:`min_pos`."""
        self.set_pos(self.min_pos)

    def jump_to_end(self):
        """Set :py:attr:`pos` equal to :py:attr:`max_pos`."""
        self.set_pos(self.max_pos)

    def grab_frame(self):
        # check difference between cv2 video pos and renderer pos
        next_frame = self.in_video.get(cv2.CAP_PROP_POS_FRAMES)
        jump = self.pos - next_frame
        # if renderer pos == cv2's next frame, then just read the next frame
        if jump == 1:
            suc, frame = self.in_video.read()
            # if frame grab was unsuccessful, then set pos back 1
            if not suc:
                self.pos -= 1
        # if not next frame, then set cv2s pos and then read
        else:
            self.in_video.set(cv2.CAP_PROP_POS_FRAMES, self.pos - 1)
            suc, frame = self.in_video.read()
            # if frame grab was unsuccessful, then grab cv2's current frame
            if not suc:
                self.pos = int(self.in_video.get(cv2.CAP_PROP_POS_FRAMES))
        return suc, frame

    def render_frame(self):
        # wait for a frame request
        self._frame_ready.wait_for_clear()
        success, frame = self.grab_frame()
        if success:
            # resize frame
            frame = self._rescale(frame)
            # if rendering is enabled, and there is data available for the current frame, draw them onto the frame
            if self.show_boxes:
                target = int(self.pos)
                if self.frame_data.has(target):
                    self.animate_frame(self.frame_data[target], frame)
            # write to out video (skips internally if out video does not exists)
            self.write_frame(frame)
            # save frame to variable so external access
            self.current_frame = frame
        # notify that frame is ready
        self._frame_ready.set()

    def animate_frame(self, frame: Frame, img: NDArray):
        if self.show_ignored:
            # draw all ignored bboxes in red without id
            for old in frame.ignored:
                self.animate_bbox(old, img, (0, 0, 255))
        if self.show_birds:
            # draw all bird bboxes in green if real or yellow if not real
            for bird in frame.birds:
                text = f"{bird.item_obj.id}.{bird.item_obj.sub_id}({bird.conf:.02f})"
                self.animate_bbox(bird, img, (0, 255, 0 if bird.real else 255), text=text, show_center=True, text_anchor="sw")
                # if rings are also shown and local_rings loaded in metrics_cache: draw lines between the bird and all rings
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
            # draw all rings bboxes in blue for plastic rings and gray for metal
            for ring in frame.rings:
                color = (255, 0, 0) if ring.item_obj.type == ItemType.RING_PLASTIC else (150, 150, 150)
                self.animate_bbox(ring, img, color)
        if self.show_events:
            # show all event bboxes in black
            for event in frame.events:
                text = f"{event.item_obj.id}.{event.item_obj.sub_id}: {event.item_obj.subject.value}"
                self.animate_bbox(event, img, (0, 0, 0), text_color=(255, 255, 255), text=text, text_anchor="se")

    def animate_bbox(
        self,
        bbox: BBox,
        img: NDArray,
        color: Color,
        text: str | None = None,
        text_color: Color = (0, 0, 0),
        text_anchor: Literal["nw", "ne", "sw", "se"] = "nw",
        show_center: bool = False,
    ):
        # rescale key variables
        x = int(bbox.x * self._scaler)
        y = int(bbox.y * self._scaler)
        xmin = int(bbox.xmin * self._scaler)
        xmax = int(bbox.xmax * self._scaler)
        ymin = int(bbox.ymin * self._scaler)
        ymax = int(bbox.ymax * self._scaler)
        thickness = int(5 * self._scaler)
        # create a point in center of bbox if requested
        if show_center:
            cv2.circle(img, (x, y), int(10 * self._scaler), color, -1)
        # add bounding box rectangle
        cv2.rectangle(img, (xmin, ymin), (xmax, ymax), color, thickness)
        # if text is provided add a label to the box
        if text:
            # ensure anchor is valid
            if text_anchor not in ("nw", "ne", "sw", "se"):
                raise ValueError(f"Unsupported text anchor ({text_anchor}) must be 'nw', 'ne', 'sw' or 'se'.")

            # get text dimensions
            (text_width, text_height), text_base = cv2.getTextSize(
                text, cv2.FONT_HERSHEY_SIMPLEX, 0.75 * self._scaler, int(2 * self._scaler)
            )

            # calculate dimensions and position of text and background frame
            text_border = int(5 * self._scaler)
            frame_width = (text_border * 2) + text_width
            frame_height = (text_border * 2) + text_height + text_base
            frame_offset = int(thickness / 2)
            frame_y = ymin + frame_offset if text_anchor[0] == "n" else ymax - frame_offset - frame_height
            frame_x = xmin + frame_offset if text_anchor[1] == "w" else xmax - frame_offset - frame_width
            text_y = frame_y + frame_offset + text_border + text_height
            text_x = frame_x + text_border

            # draw background rectangle and text on top
            cv2.rectangle(img, (frame_x, frame_y, frame_width, frame_height), color, -1)
            cv2.putText(
                img, text, (text_x, text_y), cv2.FONT_HERSHEY_SIMPLEX, 0.75 * self._scaler, text_color, int(2 * self._scaler)
            )

    def write_frame(self, frame: NDArray):
        if self.out_video:
            self.out_video.write(frame)

    def _render_loop(self):
        # the thread render loop
        while self.open:
            self.render_frame()

    def close(self):
        # mark as closed
        self.open = False

        # close open videos
        self.in_video.release()
        if self.out_video:
            self.out_video.release()

        # in case renderer thread is currently waiting for next frame, request next frame
        self._frame_ready.clear()
        # wait for thread to close
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
        scalable: bool = True,
    ):
        self.scene = scene
        self.open: bool = True
        self.mask = mask
        self.renderer = Renderer(scene.video_data.video_path, scene.frames, out_video, scaler=scale)
        self.rendered_frame = None
        self.min_sleep_time: int = int((1 / self.renderer.fps) * 1000) or 1
        self.last_render_time: float = 0
        self.paused: bool = False
        self.state: int = self.NORMAL
        self.text_entry: str = ""
        self._start: Optional[int] = None
        self._end: Optional[int] = None
        self.clipped = False
        self.layers_str: str = ""
        self._refresh_layers_str()

        window_flags = cv2.WINDOW_NORMAL if scalable else cv2.WINDOW_AUTOSIZE
        if scalable:
            cv2.namedWindow("out", window_flags)
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
        self.renderer.set_pos(val)

    def set_start(self, val: Optional[int] = None):
        self._start = val
        if self._clipped:
            self.renderer.min_pos = self._start

    def set_end(self, val: Optional[int] = None):
        self._end = val
        if self._clipped:
            self.renderer.max_pos = self._end

    def display_frames(self):
        # continues loop until animator or renders is closed
        while self.open and self.renderer.open:
            # if there is a ready frame and (the minimum sleep time has been reached or animator is paused)
            if self.renderer.frame_is_ready and (
                (time.time() - self.last_render_time) > (self.min_sleep_time / 1000) or self.paused
            ):
                # if it is a new frame, show it in the window
                if self.renderer.current_frame is not self.rendered_frame:
                    c_frame: NDArray = self.renderer.current_frame
                    cv2.imshow("out", c_frame)
                # if animator is not paused:
                if not self.paused:
                    # if render is equal to or beyond renders max_pos then pause animator
                    if self.renderer.pos >= self.renderer.max_pos:
                        self.paused = True
                    # otherwise request the next frame and log when the request was made
                    else:
                        self.renderer.pos += 1
                        self.last_render_time = time.time()
                # if animator is paused, sleep until next key command, if playing, sleep for at least min_sleep_time
                wait_time = 0 if self.paused else self.min_sleep_time
                self.update_window_name()
            else:
                wait_time = 1
            # wait until a key is pressed
            key = cv2.waitKey(wait_time)
            # choose key mappings based on if in normal or frame_jump state
            if self.state == self.NORMAL:
                self._normal_key_input(key)
            else:
                self._frame_jump_key_input(key)
            # detect if user closed the window and close animator if so
            try:
                if cv2.getWindowProperty("out", cv2.WND_PROP_VISIBLE) < 1:
                    self.close()
            except cv2.error:
                self.close()

    def _normal_key_input(self, key: int):
        match key:
            case -1:
                pass
            case 27:  # ESCAPE (quit)
                self.close()
            case 32:  # SPACE (pause/play)
                self.paused = not self.paused
                self.update_window_name()
            case 100:  # D (next fame, only if paused)
                if self.paused and self.renderer.frame_is_ready:
                    self.renderer.pos += 1
                    self.update_window_name()
            case 68:  # SHIFT + D (jump to end, only if paused)
                if self.paused and self.renderer.frame_is_ready:
                    self.renderer.jump_to_end()
            case 97:  # A (previous frame, only if paused)
                if self.paused and self.renderer.frame_is_ready:
                    self.renderer.pos -= 1
            case 65:  # SHIFT + A (jump to start, only if paused)
                if self.paused and self.renderer.frame_is_ready:
                    self.renderer.jump_to_start()
            case 101:  # E (jump forward 1 second, only if paused)
                if self.paused and self.renderer.frame_is_ready:
                    self.renderer.pos += self.renderer.fps
                    self.update_window_name()
            case 69:  # SHIFT + E (jump forward 3 seconds, only if paused)
                if self.paused and self.renderer.frame_is_ready:
                    self.renderer.pos += self.renderer.fps * 3
                    self.update_window_name()
            case 113:  # Q (jump backward 1 second, only if paused)
                if self.paused and self.renderer.frame_is_ready:
                    self.renderer.pos -= self.renderer.fps
                    self.update_window_name()
            case 81:  # SHIFT + Q (jump backward 3 seconds, only if paused)
                if self.paused and self.renderer.frame_is_ready:
                    self.renderer.pos -= self.renderer.fps * 3
                    self.update_window_name()
            case 119:  # W (increase sleep time by 1 millisecond)
                self.min_sleep_time += 1
                self.update_window_name()
            case 115:  # S (decrease sleep time by 1 millisecond)
                if self.min_sleep_time > 1:
                    self.min_sleep_time -= 1
                    self.update_window_name()
            case 106:  # J (enter jump mode, only if paused)
                if self.paused:
                    self.text_entry = "0"
                    self.state = self.FRAME_JUMP
            case 99:  # C (toggle clipped mode on/off)
                self.clipped = not self.clipped
            case 104:  # H (toggle hiding all layers)
                self.renderer.show_boxes = not self.renderer.show_boxes
                self._refresh_layers_str()
            case 49 | 156:  # 1 | numpad 1 (toggle hiding layer 1)
                self.renderer.show_ignored = not self.renderer.show_ignored
                self._refresh_layers_str()
            case 50 | 153:  # 2 | numpad 2 (toggle hiding layer 2)
                self.renderer.show_birds = not self.renderer.show_birds
                self._refresh_layers_str()
            case 51 | 155:  # 3 | numpad 3 (toggle hiding layer 3)
                self.renderer.show_rings = not self.renderer.show_rings
                self._refresh_layers_str()
            case 52 | 150:  # 4 | numpad 4 (toggle hiding layer 4)
                self.renderer.show_events = not self.renderer.show_events
                self._refresh_layers_str()
            case other:  # everything else (print that the entered key was unrecognized)
                print(f"Unrecognized key: '{chr(other)}'")

    def _frame_jump_key_input(self, key: int):
        match key:
            case 13:  # ENTER (apply jump to entered frame)
                if self.text_entry:
                    self.renderer.set_pos(int(self.text_entry))
                self.state = self.NORMAL
                self.text_entry = "0"
            case 113:  # Q (cancel jump)
                self.state = self.NORMAL
            case 8:  # BACKSPACE (delete right most digit)
                self.text_entry = self.text_entry[:-1]
                if not self.text_entry:
                    self.text_entry = "0"
            case _:  # everything else (if it's a digit, add to current frame text entry, otherwise print key was unrecognized)
                # check every digit (0-9)
                for k in range(10):
                    # check if it matches entered key
                    if key == ord(str(k)):
                        # add to entry and skip other checks
                        self.text_entry += str(k)
                        break
                # if no digit matches then print unrecognized key warning
                else:
                    print(f"Unrecognized key: '{chr(key)}'")
                    return
                # removing leading 0s if text is not equal to "0"
                if self.text_entry.startswith("0") and self.text_entry != "0":
                    self.text_entry = self.text_entry[1:]

    def update_window_name(self):
        # set window title based on animators internal state
        if self.state == self.FRAME_JUMP:
            # if in frame jump mode:
            # title: <name of video> (jump to: <user frame entry><', Clipped' if in clipped mode>) <layers selection>
            cv2.setWindowTitle(
                "out",
                f"{self.scene.video_data.name} (jump to: {self.text_entry}{", Clipped" if self.clipped else ""}) {self.layers_str}",
            )
        elif self.paused:
            # if animator is paused
            # title: <name of video> (frame: <current frame><', Clipped' if in clipped mode>) <layers selection>
            cv2.setWindowTitle(
                "out",
                f"{self.scene.video_data.name} (frame: {self.renderer.pos}{", Clipped" if self.clipped else ""}) {self.layers_str}",
            )
        else:
            # if playing:
            # title: <name of video> (sleep: <time between frames> ms<', Clipped' if in clipped mode>) <layers selection>
            cv2.setWindowTitle(
                "out",
                f"{self.scene.video_data.name} (sleep: {self.min_sleep_time} ms{", Clipped" if self.clipped else ""}) {self.layers_str}",
            )

    def close(self):
        self.open = False
        self.renderer.close()
        # ensure all windows close
        try:
            cv2.destroyWindow("out")
        except cv2.error:
            pass
