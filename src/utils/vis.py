import cv2
from numpy.typing import NDArray


def plot_bbox_on_frame(frame: NDArray, bbox: NDArray) -> NDArray:
    bbox = bbox.astype(int)
    pt1 = (bbox[0], bbox[1])
    pt2 = (bbox[2], bbox[3])
    frame = cv2.rectangle(frame, pt1, pt2, (0, 255, 0), 2)
    frame = cv2.putText(frame, str(bbox[5]), pt1, cv2.FONT_HERSHEY_COMPLEX, 3, (0, 255, 0))
    return frame


def plot_paint_center(frame, count, center):
    center = (int(center[0]), int(center[1]))
    frame = cv2.putText(
        frame,
        str(count),
        center,
        cv2.FONT_HERSHEY_SIMPLEX,
        0.8,
        (255, 0, 0),
        lineType=cv2.LINE_AA,
    )
    frame = cv2.drawMarker(
        frame,
        center,
        (255, 0, 0),
        markerType=cv2.MARKER_TILTED_CROSS,
        markerSize=15,
        thickness=3,
    )
    return frame
