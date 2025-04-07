import argparse
import os
import sys

import cv2
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.lines import Line2D
from mpl_toolkits.mplot3d import axes3d
from tqdm import tqdm

sys.path.append(".")
from src.utils import json_handler, video, yaml_handler

plt.rcParams["axes.prop_cycle"] = plt.cycler("color", plt.get_cmap("tab20").colors)
obj_cmap = plt.get_cmap("tab10")

# parser
parser = argparse.ArgumentParser()
parser.add_argument("-tc", "--th_conf", required=False, default=0.3)
parser.add_argument("-m", "--move", required=False, action="store_true")
parser.add_argument("-tm", "--th_move", required=False, default=1.0)
args = parser.parse_args()

th_conf = args.th_conf
is_move = args.move
th_move = args.th_move

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

config = yaml_handler.load("datasets/yolov8_finetuning/yolov8_finetuning.yaml")
classes = config.names.__dict__

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
        f"out/{video_name}/{video_name}_det_finetuned_thmove{th_move:.2f}.tsv",
        skiprows=1,
        dtype=float,
    )

    cap = video.Capture(f"video/{video_name}.mp4")
    frame_count = cap.frame_count
    frame_size = cap.size
    _, frame = cap.read(30 * 5)
    del cap

    fig = plt.figure(figsize=(8, 12))
    ax = fig.add_subplot(111, projection="3d")

    # show video frame on the bottom of graph
    # frame[frame < 10] = 255
    # frame = frame / 255
    # X, Y = np.mgrid[0 : frame_size[0], 0 : frame_size[1]]
    # ax.plot_surface(
    #     X,
    #     Y,
    #     np.atleast_2d(0),
    #     rstride=5,
    #     cstride=5,
    #     facecolor="white",
    #     facecolors=frame.transpose(1, 0, 2),
    #     alpha=0.1,
    # )

    # plot yolo
    for pred in tqdm(yolo_preds, ncols=100, leave=False, desc=f"{video_name} plot"):
        if is_move and pred[11] == 0:
            continue

        z, x1, y1, x2, y2 = pred[:5].astype(int)
        w, h = x2 - x1, y2 - y1
        conf = pred[5]
        if z % 10 != 0:
            continue
        if conf < th_conf:
            continue
        X, Y = np.mgrid[x1 : x2 : w - 1, y1 : y2 : h - 1]
        Z = np.full(X.shape, z)
        label = int(pred[6])
        c = obj_cmap(label)
        ax.plot_wireframe(
            X, Y, Z, edgecolor=c, facecolor="none", alpha=0.3, rstride=w, cstride=h
        )
        # ax.plot_wireframe(
        #     X, Y, Z, edgecolor="red", facecolor="white", alpha=0.1, rstride=w, cstride=h
        # )

    # plot annotation
    if len(ann_data) > 0:
        for i, label in enumerate(np.unique(ann_data.T[8])):
            tmp_data = ann_data[ann_data.T[8] == label][0]
            X = float(tmp_data[5])
            Y = float(tmp_data[6])
            Z = float(tmp_data[0])
            ax.scatter(X, Y, Z, c="black", marker="o", s=50)
            ax.text(
                X,
                Y,
                Z + 20,
                label,
                horizontalalignment="center",
                verticalalignment="bottom",
            )
            # ax.scatter(X, Y, Z, marker="o", label=label, s=50)
    else:
        tqdm.write(f"{video_name} doesn't have annotation.")

    ax.set_box_aspect((frame_size[0], frame_size[1] * 1.5, frame_size[0] * 2.0))
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.set_zlabel("n_frame")
    ax.set_xlim(0, frame_size[0])
    ax.set_ylim(0, frame_size[1])
    ax.set_zlim(0, frame_count)
    handles, labels = plt.gca().get_legend_handles_labels()
    handles += [Line2D([0], [0], color=obj_cmap(int(i))) for i in classes.keys()]
    labels += list(classes.values())
    ax.legend(handles, labels, loc="upper left", bbox_to_anchor=(0.7, 1.0))
    plt.gca().invert_xaxis()

    # img rotation
    imgs = []
    for angle in tqdm(
        range(0, 360, 5), ncols=100, leave=False, desc=f"{video_name} rotation"
    ):
        ax.view_init(elev=20, azim=angle)
        fig.canvas.draw()
        img = np.array(fig.canvas.renderer.buffer_rgba())
        img = cv2.cvtColor(img, cv2.COLOR_RGBA2BGR)
        img = img[250:-200, 150:-50]
        # cv2.imwrite(f"{video_name}_{angle}.jpg", img)
        # raise KeyError
        imgs.append(img)

    # write video
    size = imgs[0].shape[1::-1]
    out_dir = "out/plot_3d_objects"
    # out_dir = "out/annotation_3d_plot"
    os.makedirs(out_dir, exist_ok=True)
    str_move = "_move" if is_move else ""
    wrt = video.Writer(f"{out_dir}/{video_name}{str_move}.mp4", 10, size)
    wrt.write_each(imgs)
    del wrt

    plt.close()

print("complete")
