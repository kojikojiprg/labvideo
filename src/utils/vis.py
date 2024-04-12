import cv2
from numpy.typing import NDArray


def plot_bbox_on_frame(frame: NDArray, bbox: NDArray) -> NDArray:
    bbox = bbox.astype(int)
    pt1 = (bbox[0], bbox[1])
    pt2 = (bbox[2], bbox[3])
    frame = cv2.rectangle(frame, pt1, pt2, (0, 255, 0), 2)
    return frame
