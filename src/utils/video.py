import gc
import os
from typing import Optional, Tuple

import cv2
import numpy as np


class Capture:
    def __init__(self, path: str):
        if not os.path.isfile(path):
            raise ValueError(f"not exist file {path}")

        self._cap = cv2.VideoCapture(path)

        self.fps = self._cap.get(cv2.CAP_PROP_FPS)
        self.size = (
            int(self._cap.get(cv2.CAP_PROP_FRAME_WIDTH)),
            int(self._cap.get(cv2.CAP_PROP_FRAME_HEIGHT)),
        )

    def __del__(self):
        self._cap.release()
        gc.collect()

    @property
    def frame_count(self) -> int:
        # cv2.CAP_PROP_FRAME_COUNT is not correct.
        self.set_pos_frame_count(int(1e10))  # set large value
        count = int(self._cap.get(cv2.CAP_PROP_POS_FRAMES))
        self.set_pos_frame_count(0)  # initialize
        return count

    @property
    def is_opened(self) -> bool:
        return self._cap.isOpened()

    def set_pos_frame_count(self, idx: int):
        self._cap.set(cv2.CAP_PROP_POS_FRAMES, idx)

    def set_pos_frame_time(self, begin_sec: int):
        self._cap.set(cv2.CAP_PROP_POS_FRAMES, begin_sec * self.fps)

    def read(self, idx: Optional[int] = None) -> Tuple[bool, Optional[np.array]]:
        if idx is not None:
            self.set_pos_frame_count(idx)

        return self._cap.read()


class Writer:
    def __init__(self, path, fps, size, fmt="mp4v"):
        out_dir = os.path.dirname(path)
        if not os.path.exists(out_dir):
            os.makedirs(out_dir, exist_ok=True)

        # writer object
        fmt = cv2.VideoWriter_fourcc(*fmt)
        self._writer = cv2.VideoWriter(path, fmt, fps, size)

    def __del__(self):
        self._writer.release()
        gc.collect()

    def write(self, frame):
        self._writer.write(frame)

    def write_each(self, frames):
        for frame in frames:
            self._writer.write(frame)
