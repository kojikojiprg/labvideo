import os
import sys

import matplotlib.pyplot as plt
import numpy as np
from tqdm import tqdm

sys.path.append(".")
from src.utils import json_handler, yaml_handler

TH_MOVE = 4.55
MAX_N_COUNTS = 5

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


def create_legend(max_n_counts, marker_scale=100):
    counts = np.arange(max_n_counts + 1)
    fig = plt.figure()
    ax = fig.subplots(1, 1)
    for c in counts:
        if c < max_n_counts:
            label = c
        else:
            label = f"{c}~"
        ax.scatter(
            c,
            c,
            c * marker_scale,
            marker="|",
            linewidth=c / 5,
            color="black",
            label=label,
        )
    hans, labs = ax.get_legend_handles_labels()
    plt.close()

    return hans, labs


def plot_timing(
    counts_dict, ann_timings, fig_path, max_n_counts=MAX_N_COUNTS, marker_scale=100
):
    fig = plt.figure(figsize=(12, 6))
    ax = fig.subplots(1, 1)

    # plot object counts
    for label, counts in counts_dict.items():
        y = np.full((len(counts),), len(classes) - (label + 1))
        x = np.arange(len(counts))
        counts = np.array(counts).astype(float)
        counts[counts > max_n_counts] = max_n_counts
        s = counts * marker_scale
        lw = (counts + 1) / (max_n_counts + 1)
        ax.scatter(x, y, label=classes[str(label)], marker="|", s=s, linewidths=lw)

    # plot annotation timing
    for label, n_frame in ann_timings.items():
        ax.vlines([n_frame], -0.2, len(classes) - 1 + 0.2, color="black", lw=0.5)
        ax.text(n_frame, len(classes) - 1 + 0.25, label, horizontalalignment="center")

    ax.set_xlabel("number of frames")
    ticks = [len(classes) - (int(k) + 1) for k in classes.keys()]
    ax.set_yticks(ticks, list(classes.values()))

    hans, labs = create_legend(max_n_counts, marker_scale)
    ax.legend(
        hans,
        labs,
        loc="upper left",
        bbox_to_anchor=(1.0, 1.0),
        borderpad=1.0,
        labelspacing=1.5,
    )

    plt.savefig(fig_path, bbox_inches="tight")
    plt.close()


def plot_histogram(counts_dict, classes, fig_path, max_n_counts=MAX_N_COUNTS):
    labels = list(classes.values())
    cmap = plt.get_cmap("tab10")

    fig, axs = plt.subplots(1, len(labels), figsize=(22, 4))
    for i, label in enumerate(labels):
        counts_hist = np.array(counts_dict[i])
        counts_hist[counts_hist > max_n_counts] = max_n_counts

        c = cmap(i)

        axs[i].hist(
            counts_hist,
            bins=max_n_counts + 1,
            range=(0, max_n_counts + 1),
            color=c,
            label=label,
        )
        axs[i].set_xlabel("number of counts")
        axs[i].set_xticks(
            [0.5 + i for i in range(max_n_counts + 1)], list(range(max_n_counts + 1))
        )
        axs[i].set_title(label)
    axs[0].set_ylabel("number of frames")
    plt.savefig(fig_path, bbox_inches="tight")
    plt.close()


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
    ann_data = np.loadtxt(
        f"out/{video_name}/{video_name}_ann.tsv", skiprows=1, dtype=str
    )
    yolo_preds = np.loadtxt(
        f"out/{video_name}/{video_name}_det_finetuned_thmove{TH_MOVE}.tsv",
        skiprows=1,
        dtype=float,
    )

    if len(ann_data) == 0:
        tqdm.write(f"{video_name} doesn't have annotation.")
        continue

    # get annotaion timing
    ann_unique_labels = np.unique(ann_data[:, 8])
    ann_timings = {ann_label: 0 for ann_label in ann_unique_labels}
    for ann_label in ann_unique_labels:
        ann_tmp = ann_data[ann_data[:, 8] == ann_label]
        n_frame = int(ann_tmp[0, 0])
        ann_timings[ann_label] = n_frame

    # count object
    last_n_frame = int(np.max(yolo_preds[:, 0]))
    counts_dict = {
        int(label): [0 for n_frame in range(last_n_frame + 1)]
        for label in classes.keys()
    }
    counts_dict_move = {
        int(label): [0 for n_frame in range(last_n_frame + 1)]
        for label in classes.keys()
    }
    for pred in yolo_preds:
        n_frame = int(pred[0])
        label = int(pred[6])
        is_move = bool(pred[-1])

        counts_dict[label][n_frame] += 1
        if is_move:
            counts_dict_move[label][n_frame] += 1

    out_dir = "out/object_detection_analysis/timing"
    os.makedirs(out_dir, exist_ok=True)
    fig_path = f"{out_dir}/{video_name}_object_timing.png"
    plot_timing(counts_dict, ann_timings, fig_path, MAX_N_COUNTS)
    fig_path = f"{out_dir}/{video_name}_object_timing_move.png"
    plot_timing(counts_dict_move, ann_timings, fig_path, MAX_N_COUNTS)

    # histogram
    out_dir = "out/object_detection_analysis/histogram"
    os.makedirs(out_dir, exist_ok=True)
    fig_path = f"{out_dir}/{video_name}_object_hist.png"
    plot_histogram(counts_dict, classes, fig_path, MAX_N_COUNTS)
    fig_path = f"{out_dir}/{video_name}_object_hist_move.png"
    plot_histogram(counts_dict_move, classes, fig_path, MAX_N_COUNTS)

print("complete")
