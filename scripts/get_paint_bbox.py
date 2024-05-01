import sys

import cv2
import numpy as np
from tqdm import tqdm

sys.path.append("src")
from utils import json_handler, video

ann_json_path = "annotation/annotation.json"
ann_json = json_handler.load(ann_json_path)
# info_json_path = "annotation/info.json"
# info_json = json_handler.load(info_json_path)

paint_bbox_json = {}
error_paint_videos = [("video_id", "aid", "error")]
for video_id, ann_lst in tqdm(ann_json.items(), ncols=100):
    # if video_id not in info_json:
    #     tqdm.write(f"miss {video_id}")
    #     continue
    paint_bbox_json[video_id] = []
    for ann in tqdm(ann_lst, ncols=100, leave=False):
        if ann["type"] != "Paint":
            continue

        # open paint video
        aid = ann["aid"]
        paint_video_path = f"annotation/video/{aid}.mp4"
        cap = video.Capture(paint_video_path)
        if not cap.is_opened:
            error_paint_videos.append((video_id, aid, "Couldn't open paint video"))
            paint_bbox_json[video_id].append({"aid": aid, "bbox": None, "center": None})
            continue

        # read first and last frames
        _, first_frame = cap.read(0)
        _, last_frame = cap.read(cap.frame_count - 1)
        del cap

        # gray scale
        first_frame_gray = cv2.cvtColor(first_frame, cv2.COLOR_RGB2GRAY)
        last_frame_gray = cv2.cvtColor(last_frame, cv2.COLOR_RGB2GRAY)

        # 余白部分を255にする
        last_frame_gray[first_frame_gray < 10] = 255

        # binary
        _, last_frame_bin = cv2.threshold(
            last_frame_gray, 10, 255, cv2.THRESH_BINARY_INV
        )

        # find contours
        contours, hierarchy = cv2.findContours(
            last_frame_bin, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
        )
        contours = list(
            filter(
                lambda x: 100 < cv2.contourArea(x) and cv2.contourArea(x) < 1200 * 700,
                contours,
            )
        )
        if len(contours) == 0:
            error_paint_videos.append((video_id, aid, "No painted circle"))
            paint_bbox_json[video_id].append({"aid": aid, "bbox": None, "center": None})
            continue

        # detect bboxs
        for contour in contours:
            x, y, w, h = cv2.boundingRect(contour)
            if w < 20 or h < 20:
                continue  # skip thin objects
            paint_bbox_json[video_id].append(
                {
                    "aid": aid,
                    "bbox": (x, y, x + w, y + h),
                    "center": (x + w / 2, y + h / 2),
                }
            )

# save data
json_handler.dump("annotation/paint_bbox.json", paint_bbox_json)
np.savetxt("annotation/paint_error.tsv", error_paint_videos, "%s", "\t")
