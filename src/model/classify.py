import os
import shutil
from glob import glob

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import (
    ConfusionMatrixDisplay,
    classification_report,
    confusion_matrix,
)
from tqdm import tqdm
from ultralytics import YOLO


def train_classify(data_name, data_type, split_type, epochs=100):
    model = YOLO("models/yolo/yolov8n-cls.pt")
    model.train(
        data=f"classify/{split_type}/{data_name}/{data_type}",
        epochs=epochs,
        task="classify",
    )

    # get trained data dir
    dirs = sorted(glob("runs/classify/train*/"))
    trained_dir = dirs[-1]

    yolo_result_dir = f"runs/classify/{split_type}/{data_name}/{data_type}"
    if os.path.exists(yolo_result_dir):
        dirs = sorted(glob(yolo_result_dir + "-v*/"))
        if len(dirs) == 0:
            v_num = 1
        else:
            last_dir = dirs[-1]
            v_num = int(os.path.dirname(last_dir).split("-")[-1].replace("v", "")) + 1
        yolo_result_dir = f"runs/classify/{data_name}/{data_type}-v{v_num}"
        shutil.move(trained_dir, yolo_result_dir)
    else:
        shutil.move(trained_dir, yolo_result_dir)
    os.makedirs(yolo_result_dir, exist_ok=True)

    return yolo_result_dir


def pred_classify(img_paths, stage, yolo_result_dir, data_type):
    weights_path = os.path.join(yolo_result_dir, "weights", "last.pt")
    model = YOLO(weights_path)

    results = []
    missed_img_paths = []
    for path in tqdm(img_paths):
        label = os.path.basename(os.path.dirname(path))
        pred = model(path)
        pred_label_id = pred[0].probs.top1
        names = pred[0].names
        pred_label = names[pred_label_id]
        results.append([label, pred_label])
        if label != pred_label:
            missed_img_paths.append([path, label, pred_label])

    results = np.array(results)
    cm = confusion_matrix(results.T[0], results.T[1])
    path = f"{yolo_result_dir}/cm_{stage}_num.jpg"
    cm_plot(cm, path, data_type)
    cm = confusion_matrix(results.T[0], results.T[1], normalize="true")
    path = f"{yolo_result_dir}/cm_{stage}_recall.jpg"
    cm_plot(cm, path, data_type)
    cm = confusion_matrix(results.T[0], results.T[1], normalize="pred")
    path = f"{yolo_result_dir}/cm_{stage}_precision.jpg"
    cm_plot(cm, path, data_type)
    cm = confusion_matrix(results.T[0], results.T[1], normalize="all")
    path = f"{yolo_result_dir}/cm_{stage}_f1.jpg"
    cm_plot(cm, path, data_type)

    path = f"{yolo_result_dir}/cm_{stage}_report.tsv"
    report = classification_report(
        results.T[0], results.T[1], digits=3, output_dict=True, zero_division=0
    )
    pd.DataFrame.from_dict(report).T.to_csv(path, sep="\t")
    return results, missed_img_paths


def cm_plot(cm, path, data_type):
    cmd = ConfusionMatrixDisplay(cm)
    cmd.plot(
        xticks_rotation="vertical",
        include_values=data_type == "label_type",
        cmap="Blues",
    )
    plt.xticks(fontsize=6)
    plt.yticks(fontsize=6)
    plt.savefig(path, bbox_inches="tight")
    plt.close()
