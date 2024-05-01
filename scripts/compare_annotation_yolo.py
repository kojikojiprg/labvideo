import os
import sys

import cv2
import matplotlib.pyplot as plt
import numpy as np
from mpl_toolkits.mplot3d import axes3d
from tqdm import tqdm

sys.path.append("src")
from utils import json_handler, video

plt.rcParams["axes.prop_cycle"] = plt.cycler("color", plt.get_cmap("tab20").colors)

# params
th_size = 20000
th_conf = 0.3

ann_json = json_handler.load("annotation/annotation.json")
info_json = json_handler.load("annotation/info.json")

video_id_to_name = {
    data[0]: data[1].split(".")[0]
    for data in np.loadtxt(
        "annotation/annotation.tsv", str, delimiter="\t", skiprows=1, usecols=[1, 2]
    )
    if data[0] != "" and data[1] != ""
}

for video_id, ann_lst in tqdm(ann_json.items(), ncols=100):
    if video_id not in info_json:
        tqdm.write(f"{video_id} is not in info.json")
        continue

    video_name = video_id_to_name[video_id]

    if not os.path.exists(f"out/{video_name}/{video_name}_ann.tsv"):
        tqdm.write(f"{video_name} doesn't have annotation")
        continue
    tqdm.write(f"loading {video_name} annotation")
    ann_data = np.loadtxt(
        f"out/{video_name}/{video_name}_ann.tsv", skiprows=1, dtype=str
    )
    yolo_preds = np.loadtxt(
        f"out/{video_name}/{video_name}_yolo.tsv", skiprows=1, dtype=float
    )

    cap = video.Capture(f"video/{video_name}.mp4")
    frame_count = cap.frame_count
    frame_size = cap.size
    _, frame = cap.read(30 * 5)
    del cap

    fig = plt.figure(figsize=(8, 12))
    ax = fig.add_subplot(111, projection="3d")

    # plot yolo
    for pred in tqdm(yolo_preds, ncols=100, leave=False, desc=f"{video_name} plot"):
        z, x1, y1, x2, y2 = pred[:5].astype(int)
        w, h = x2 - x1, y2 - y1
        conf = pred[5]
        if z % 10 != 0:
            continue
        if w * h > th_size:
            continue
        if conf < th_conf:
            continue
        X, Y = np.mgrid[x1 : x2 : w - 1, y1 : y2 : h - 1]
        Z = np.full(X.shape, z)
        ax.plot_wireframe(
            X, Y, Z, edgecolor="red", facecolor="white", alpha=0.1, rstride=w, cstride=h
        )

    # plot annotation
    if len(ann_data) > 0:
        for i, label in enumerate(np.unique(ann_data.T[8])):
            tmp_data = ann_data[np.where(ann_data.T[8] == label)[0]]
            X = tmp_data.T[5].astype(float)
            Y = tmp_data.T[6].astype(float)
            Z = tmp_data.T[0].astype(int)
            ax.scatter(X, Y, Z, marker="o", label=label, s=10, alpha=0.9)
    else:
        tqdm.write(f"{video_name} doesn't have annotation.")

    # show vide oframe on the bottom of graph
    frame = cv2.cvtColor(frame, cv2.COLOR_RGB2RGBA)
    frame = frame / 255
    X, Y = np.mgrid[0 : frame_size[0], 0 : frame_size[1]]
    ax.plot_surface(
        X,
        Y,
        np.atleast_2d(0),
        rstride=10,
        cstride=10,
        facecolors=frame.transpose(1, 0, 2),
    )

    ax.set_box_aspect((frame_size[0], frame_size[1] * 1.5, frame_size[0] * 2.0))
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.set_zlabel("n_frame")
    ax.set_xlim(0, frame_size[0])
    ax.set_ylim(0, frame_size[1])
    ax.set_zlim(0, frame_count)
    ax.legend()
    plt.gca().invert_xaxis()

    # img rotation
    imgs = []
    for angle in tqdm(
        range(0, 360, 5), ncols=100, leave=False, desc=f"{video_name} rotation"
    ):
        ax.view_init(elev=30, azim=angle)
        fig.canvas.draw()
        img = np.array(fig.canvas.renderer.buffer_rgba())
        img = cv2.cvtColor(img, cv2.COLOR_RGBA2RGB)
        img = img[250:-200, 150:-50]
        imgs.append(img)

    # write video
    size = imgs[0].shape[1::-1]
    # wrt = video.Writer(f"out/{video_name}/{video_name}_plot.mp4", 30, size)
    out_dir = "out/compare_annotation_yolo"
    os.makedirs(out_dir, exist_ok=True)
    wrt = video.Writer(f"{out_dir}/{video_name}_plot.mp4", 30, size)
    wrt.write_each(imgs)
    del wrt

    plt.close()
