import os
import sys
import urllib.request
from typing import List

import numpy as np
from ultralytics import YOLO

sys.path.append("src")
from utils import yaml_handler


class ObjectTracking:
    def __init__(self, cfg_path: dict, device: str):
        self._cfg = yaml_handler.load(cfg_path)
        self._device = device

        yolo_url = self._cfg.yolo
        yolo_name = os.path.basename(yolo_url)
        yolo_path = f"./models/yolo/{yolo_name}"
        self._download_weights(yolo_url, yolo_path)
        self._yolo = YOLO(yolo_path, verbose=False).to(self._device)

    def __del__(self):
        del self._det_model

    @staticmethod
    def _download_weights(url, path):
        if not os.path.exists(path):
            print(f"downloading model wights from {url}")
            os.makedirs(os.path.dirname(path), exist_ok=True)
            urllib.request.urlretrieve(url, path)

    def predict(self, img: np.array):
        bboxes = self._yolo.predict(img, verbose=False)[0].boxes.data.cpu().numpy()
        bboxes = bboxes[bboxes[:, 4] > self._cfg.th_conf]
        bboxes = bboxes[nms(bboxes, self._cfg.th_iou)]

        return bboxes


def nms(dets: np.ndarray, thr: float) -> List[int]:
    """Greedily select boxes with high confidence and overlap <= thr.

    Args:
        dets (np.ndarray): [[x1, y1, x2, y2, score]].
        thr (float): Retain overlap < thr.

    Returns:
        list: Indexes to keep.
    """
    if len(dets) == 0:
        return []

    x1 = dets[:, 0]
    y1 = dets[:, 1]
    x2 = dets[:, 2]
    y2 = dets[:, 3]
    scores = dets[:, 4]

    areas = (x2 - x1 + 1) * (y2 - y1 + 1)
    order = scores.argsort()[::-1]

    keep = []
    while len(order) > 0:
        i = order[0]
        keep.append(i)
        xx1 = np.maximum(x1[i], x1[order[1:]])
        yy1 = np.maximum(y1[i], y1[order[1:]])
        xx2 = np.minimum(x2[i], x2[order[1:]])
        yy2 = np.minimum(y2[i], y2[order[1:]])

        w = np.maximum(0.0, xx2 - xx1 + 1)
        h = np.maximum(0.0, yy2 - yy1 + 1)
        inter = w * h
        ovr = inter / (areas[i] + areas[order[1:]] - inter)

        inds = np.where(ovr <= thr)[0]
        order = order[inds + 1]

    return keep
