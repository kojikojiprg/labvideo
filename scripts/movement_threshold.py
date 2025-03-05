import os
import sys

import numpy as np
from tqdm import tqdm

sys.path.append(".")
from src.utils import json_handler, video, vis

# load json files
ann_json = json_handler.load("annotation/annotation.json")
info_json = json_handler.load("annotation/info.json")

video_id_to_name = {
    data[0]: data[1].split(".")[0]
    for data in np.loadtxt(
        "annotation/annotation.tsv", str, delimiter="\t", skiprows=1, usecols=[1, 2]
    )
    if data[0] != "" and data[1] != ""
}

print("calculate diffferences of object movement")
yolo_preds_all = {}
bbox_diffs = []
for video_id, ann_lst in tqdm(ann_json.items(), ncols=100):
    if video_id not in info_json:
        tqdm.write(f"{video_id} is not in info.json")
        continue

    video_name = video_id_to_name[video_id]

    if not os.path.exists(f"out/{video_name}/{video_name}_ann.tsv"):
        tqdm.write(f"{video_name} doesn't have annotation")
        continue
    tqdm.write(f"loading {video_name}")
    yolo_preds = np.loadtxt(
        f"out/{video_name}/{video_name}_det_finetuned.tsv", skiprows=1, dtype=float
    )

    yolo_preds_append_diff = np.hstack([yolo_preds, np.zeros((len(yolo_preds), 2))])
    pre_objects = {}
    for i, pred in enumerate(tqdm(yolo_preds_append_diff, ncols=100, leave=False)):
        n_frame = int(pred[0])
        bbox = pred[1:5].astype(np.float32)
        label = int(pred[6])
        tid = int(pred[7])

        if tid not in pre_objects:
            pre_objects[tid] = (n_frame, bbox, label)
            continue

        if n_frame - pre_objects[tid][0] > 1:
            del pre_objects[tid]
            continue
        if label != pre_objects[tid][2]:
            del pre_objects[tid]
            continue

        diff = bbox - pre_objects[tid][1]
        x1, y1, x2, y2 = diff
        d1 = np.sqrt(x1**2 + y1**2)
        d2 = np.sqrt(x2**2 + y2**2)
        d = (d1 + d2) / 2

        bbox_diffs.append(d)
        yolo_preds_append_diff[i, -2] = d

        pre_objects[tid] = (n_frame, bbox, label)

    yolo_preds_all[video_id] = yolo_preds_append_diff


def otsu_score(data, thresh):
    w_0 = np.sum(data <= thresh) / data.shape[0]
    w_1 = np.sum(data > thresh) / data.shape[0]
    # check ideal case
    if (w_0 == 0) | (w_1 == 0):
        return 0
    mean_all = data.mean()
    mean_0 = data[data <= thresh].mean()
    mean_1 = data[data > thresh].mean()
    sigma2_b = w_0 * ((mean_0 - mean_all) ** 2) + w_1 * ((mean_1 - mean_all) ** 2)

    return sigma2_b


bbox_diffs = np.array(bbox_diffs)

max_calc_value = 10
freq = 100
scores_otsu = np.zeros(max_calc_value * freq)
for i in tqdm(range(len(scores_otsu)), ncols=100):
    scores_otsu[i] = otsu_score(bbox_diffs, i / freq)
thresh_otsu = np.argmax(scores_otsu) / freq
print("thresholds", thresh_otsu)

print("saving tsv")
for video_id, yolo_preds_append_diff in tqdm(yolo_preds_all.items(), ncols=100):
    video_name = video_id_to_name[video_id]
    tqdm.write(f"writing {video_name}")

    for i, pred in enumerate(tqdm(yolo_preds_append_diff, ncols=100, leave=False)):
        yolo_preds_append_diff[i, -1] = int(pred[-2] >= thresh_otsu)

    # save tsv
    out_dir = os.path.join("out", video_name)
    header = "n_frame\tx1\ty1\tx2\ty2\tconf\tcls\ttid\tstart_frame\ttracklet_len\tdiff\tis_move"
    fmt = ("%d", "%f", "%f", "%f", "%f", "%f", "%d", "%d", "%d", "%d", "%f", "%d")
    det_tsv_path = os.path.join(out_dir, f"{video_name}_det_finetuned_thmove{thresh_otsu:.2f}.tsv")
    np.savetxt(det_tsv_path, yolo_preds_append_diff, fmt, "\t", header=header, comments="")

print("writing videos")
for video_id, yolo_preds_append_diff in tqdm(yolo_preds_all.items(), ncols=100):
    video_name = video_id_to_name[video_id]
    tqdm.write(f"writing {video_name}")

    # oepn video
    video_path = f"video/{video_name}.mp4"
    cap = video.Capture(video_path)

    # create video writer
    out_dir = os.path.join("out", video_name)
    yolo_video_path = os.path.join(
        out_dir, f"{video_name}_det_finetuned_thmove{thresh_otsu:.2f}.mp4"
    )
    wtr = video.Writer(yolo_video_path, cap.fps, cap.size)

    # writing yolo preds
    for n_frame in tqdm(range(cap.frame_count), ncols=100, leave=False):
        ret, frame = cap.read()
        if np.all(frame < 10):
            continue  # skip black frame

        yolo_preds_tmp = yolo_preds_append_diff[
            yolo_preds_append_diff.T[0].astype(int) == n_frame
        ]
        for pred in yolo_preds_tmp:
            d = pred[-2]
            if d < thresh_otsu:
                thickness = 1
            else:
                thickness = 3
            frame = vis.plot_bbox_on_frame(frame, pred[1:], thickness)
        wtr.write(frame)

    del cap, wtr

print("complete")
