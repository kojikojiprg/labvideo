import os
import sys
from glob import glob

import numpy as np
from tqdm import tqdm

sys.path.append("src")
from model import YOLO
from utils import video, vis

# create model
config_path = "configs/yolo.yaml"
model = YOLO(config_path, "cuda")

# get all video path
video_paths = glob(os.path.join("video", "*.mp4"))

for video_path in tqdm(video_paths, ncols=100):
    # oepn video
    cap = video.Capture(video_path)

    # create outpu directory
    video_name = os.path.basename(video_path).split(".")[0]
    out_dir = os.path.join("out", video_name)
    os.makedirs(out_dir, exist_ok=True)

    # create video writer
    yolo_video_path = os.path.join(out_dir, f"{video_name}_yolo.mp4")
    wtr = video.Writer(yolo_video_path, cap.fps, cap.size)

    # predict using yolo
    yolo_rslt = []
    for frame_num in tqdm(
        range(cap.frame_count), ncols=100, leave=False, desc=video_name
    ):
        ret, frame = cap.read()
        bboxs = model.predict(frame)
        for bbox in bboxs:
            frame = vis.plot_bbox_on_frame(frame, bbox)
            yolo_rslt.append([frame_num] + bbox.tolist())
        wtr.write(frame)

    # save tsv
    header = "n_frame\tx1\ty1\tx2\ty2\tconf\tclass"
    fmt = ("%d", "%f", "%f", "%f", "%f", "%f", "%d")
    yolo_tsv_path = os.path.join(out_dir, f"{video_name}_json.tsv")
    np.savetxt(yolo_tsv_path, yolo_rslt, fmt, "\t", header=header)

    del cap, wtr, yolo_rslt