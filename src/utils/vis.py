import cv2
import matplotlib.pyplot as plt
import numpy as np
import seaborn
from numpy.typing import NDArray

cmap = plt.get_cmap("tab10")


def plot_bbox_on_frame(frame: NDArray, bbox: NDArray) -> NDArray:
    x1, y1, x2, y2 = bbox[:4].astype(int)
    pt1 = (x1, y1)
    pt2 = (x2, y2)
    pt_center = (x1 + (x2 - x1) // 2, y1 + (y2 - y1) // 2)
    # score = float(bbox[4])
    label = int(bbox[5])
    _id = int(bbox[6])

    # get color
    c = (np.array(cmap(label)[:3]) * 255).astype(int).tolist()
    c = tuple(c[::-1])  # RGB -> BGR

    # bbox
    frame = cv2.rectangle(frame, pt1, pt2, c, 2)

    # label
    frame = cv2.putText(frame, str(label), pt1, cv2.FONT_HERSHEY_COMPLEX, 0.7, c, 1)

    # tracking ID
    frame = cv2.putText(
        frame, str(_id), pt_center, cv2.FONT_HERSHEY_COMPLEX, 0.5, (0, 0, 0), 1
    )
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


def plot_cm(cm, labels, save_path, normalized=False, on_plot=True):
    array = cm.copy()
    array[array < 0.005] = np.nan  # don't annotate (would appear as 0.00)

    fig, ax = plt.subplots(1, 1, figsize=(12, 9), tight_layout=True)
    ticklabels = labels
    seaborn.heatmap(
        array,
        ax=ax,
        annot=True,
        annot_kws={"size": 8},
        cmap="Blues",
        fmt=".2f" if normalized else ".0f",
        square=True,
        vmin=0.0,
        xticklabels=ticklabels,
        yticklabels=ticklabels,
    ).set_facecolor((1, 1, 1))

    # title = "Confusion Matrix" + " Normalized" * normalize
    # ax.set_title(title)
    ax.set_xlabel("True")
    ax.set_xticklabels(labels, rotation=0)
    ax.set_ylabel("Predicted")
    ax.set_yticklabels(labels, rotation=0)
    fig.savefig(save_path, dpi=250)
    if on_plot:
        plt.show()
    plt.close(fig)
