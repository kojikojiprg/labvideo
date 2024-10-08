import argparse
import os
import sys
from glob import glob

import numpy as np
from tqdm import tqdm

sys.path.append(".")
from src.model import ObjectTracking
from src.utils import video, vis

parser = argparse.ArgumentParser()
parser.add_argument("-f", "--finetuned_model", action="store_true")
parser.add_argument("-v", "--video", action="store_true")
args = parser.parse_args()

use_finetuned_model = args.finetuned_model
if use_finetuned_model:
    config_path = "configs/object_detection_finetuned.yml"
else:
    config_path = "configs/object_detection.yml"

# get all video path
video_paths = sorted(glob(os.path.join("video", "*.mp4")))

for video_path in tqdm(video_paths, ncols=100):
    # create model
    model = ObjectTracking(config_path, "cuda")

    # oepn video
    cap = video.Capture(video_path)

    # create outpu directory
    video_name = os.path.basename(video_path).split(".")[0]
    out_dir = os.path.join("out", video_name)
    os.makedirs(out_dir, exist_ok=True)

    # create video writer
    if args.video:
        if use_finetuned_model:
            yolo_video_path = os.path.join(out_dir, f"{video_name}_det_finetuned.mp4")
        else:
            yolo_video_path = os.path.join(out_dir, f"{video_name}_det.mp4")
        wtr = video.Writer(yolo_video_path, cap.fps, cap.size)

    # predict using yolo
    det_rslt = []
    for frame_num in tqdm(
        range(cap.frame_count), ncols=100, leave=False, desc=video_name
    ):
        ret, frame = cap.read()
        if np.all(frame < 10):
            continue  # skip black frame
        bboxs = model.predict(frame)
        for bbox in bboxs:
            bbox = np.array(bbox)
            det_rslt.append([frame_num] + bbox.tolist())
            if args.video:
                frame = vis.plot_bbox_on_frame(frame, bbox)
        if args.video:
            wtr.write(frame)

    # save tsv
    header = "n_frame\tx1\ty1\tx2\ty2\tconf\tcls\ttid\tstart_frame\ttracklet_len"
    fmt = ("%d", "%f", "%f", "%f", "%f", "%f", "%d", "%d", "%d", "%d")
    if use_finetuned_model:
        det_tsv_path = os.path.join(out_dir, f"{video_name}_det_finetuned.tsv")
    else:
        det_tsv_path = os.path.join(out_dir, f"{video_name}_det.tsv")
    np.savetxt(det_tsv_path, det_rslt, fmt, "\t", header=header, comments="")

    del model  # reset model
    del cap, det_rslt
    if args.video:
        del wtr
