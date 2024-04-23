import argparse
import sys

import numpy as np
from tqdm import tqdm

sys.path.append("src")
from utils import json_handler, video, vis

parser = argparse.ArgumentParser()
parser.add_argument("-v", "--video", action="store_true")
args = parser.parse_args()

ann_json = json_handler.load("annotation/annotation.json")
info_json = json_handler.load("annotation/info.json")
paint_data_json = json_handler.load("annotation/paint_bbox.json")

video_id_to_name = {
    data[0]: data[1].split(".")[0]
    for data in np.loadtxt(
        "annotation/annotation.tsv", str, delimiter="\t", skiprows=1, usecols=[1, 2]
    )
    if data[0] != "" and data[1] != ""
}


def _get_paint_ann(ann_list, reply):
    for ann in ann_list:
        if ann["type"] == "Paint":
            if ann["aid"] == reply:
                return ann


for video_id, ann_lst in tqdm(ann_json.items(), ncols=100):
    if video_id not in info_json:
        tqdm.write(f"{video_id} is not in info.json")
        continue

    video_name = video_id_to_name[video_id]
    paint_data = paint_data_json[video_id]

    # collect annotation data
    ann_dict = {}
    for ann in ann_lst:
        if ann["type"] == "Paint":
            continue

        comment = ann["text"]
        if "終了" not in comment:
            # get start time
            count = int(comment.split("(")[0])
            start_time = float(ann["time"])

            # get paint annotation
            reply = ann["reply"]
            paint_ann = _get_paint_ann(ann_lst, reply)
            if paint_ann is None:
                tqdm.write(f"Paint data {count} of {video_name} couldn't be found.")
                continue
            paint_aid = paint_ann["aid"]
            paint_text = paint_ann["text"]
            paint_bbox = [
                data["bbox"] for data in paint_data if data["aid"] == paint_aid
            ]
            paint_center = [
                data["center"] for data in paint_data if data["aid"] == paint_aid
            ]

            # create annotation dict
            ann_dict[count] = {
                "start_time": start_time,
                "end_time": -1.0,
                "label": comment,
                "comment": paint_text,
                "bbox": paint_bbox,
                "center": paint_center,
            }
        else:
            # get end time
            count_str = comment.split("終了")[0]
            for count in count_str.split(","):
                count = int(count)
                if count in ann_dict:
                    end_time = float(ann["time"])
                    ann_dict[count]["end_time"] = end_time

    cap = video.Capture(f"video/{video_name}.mp4")

    # create time series data
    ann_tsv = []
    for count, ann in ann_dict.items():
        label = ann["label"]
        comment = ann["comment"]
        start_frame = int(ann["start_time"] * cap.fps)
        if int(ann["end_time"]) > 0:
            end_frame = int(ann["end_time"] * cap.fps)
        else:
            end_frame = cap.frame_count

        for n_frame in range(start_frame, end_frame + 1):
            for bbox, center in zip(ann["bbox"], ann["center"]):
                x1, y1, x2, y2 = bbox
                xc, yc = center
                ann_tsv.append([n_frame, x1, y1, x2, y2, xc, yc, count, label, comment])

    ann_tsv = sorted(ann_tsv, key=lambda x: x[0])
    header = "n_frame\tx1\ty1\tx2\ty2\txc\tyc\tcount\tlabel\tcomment"
    np.savetxt(
        f"out/{video_name}/{video_name}_ann.tsv",
        ann_tsv,
        "%s",
        "\t",
        header=header,
        comments="",
    )

    # write video
    if not args.video:
        continue  # skip writing video
    wrt = video.Writer(f"out/{video_name}/{video_name}_ann.mp4", cap.fps, cap.size)
    for n_frame in tqdm(
        range(cap.frame_count), ncols=100, leave=False, desc=video_name
    ):
        ret, frame = cap.read()

        # plot annotation
        ann_tmp_list = [ann for ann in ann_tsv if ann[0] == n_frame]
        for ann_tmp in ann_tmp_list:
            center = (ann_tmp[5], ann_tmp[6])
            count = ann_tmp[7]
            frame = vis.plot_paint_center(frame, count, center)

        wrt.write(frame)
    del cap, wrt
