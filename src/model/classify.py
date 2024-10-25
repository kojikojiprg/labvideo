import os

import numpy as np
import pandas as pd
from sklearn.metrics import classification_report
from tqdm import tqdm
from ultralytics import YOLO

from src.utils import vis


def train_classify(data_name, data_type, split_type, epochs=100, batch_size=1024):
    yolo_result_dir = f"runs/classify/{data_name}/{split_type}/{data_type}"

    model = YOLO("models/yolo/yolov8n-cls.pt")
    model = model.cuda()
    model.train(
        data=f"classify/{data_name}/{split_type}/{data_type}",
        name=yolo_result_dir.replace("runs/classify/", ""),
        task="classify",
        epochs=epochs,
        batch=batch_size,
        optimizer="Adam",
        lr0=0.001,
        lrf=0.001,  # not schedule learning rate
        momentum=0.9,  # momentum is beta1 (NOTE: beta2 is fixed to 0.999)
        weight_decay=0.0,
    )

    return yolo_result_dir


def pred_classify(img_paths, stage, yolo_result_dir, labels):
    weights_path = os.path.join(yolo_result_dir, "weights", "last.pt")
    model = YOLO(weights_path)

    results = []
    missed_img_paths = []
    cm = np.zeros((len(labels), len(labels)))
    label2idx = {label: i for i, label in enumerate(labels)}
    for path in tqdm(img_paths):
        # extract ground truth
        true_label = os.path.basename(os.path.dirname(path))

        # prediction
        pred = model(path)
        pred_label_id = pred[0].probs.top1
        names = pred[0].names
        pred_label = names[pred_label_id]

        results.append([true_label, pred_label])

        true_idx = label2idx[true_label]
        pred_idx = label2idx[pred_label]
        cm[pred_idx, true_idx] += 1
        if true_label != pred_label:
            missed_img_paths.append([path, true_label, pred_label])

    results = sorted(results, key=lambda x: x[0])
    results = np.array(results)

    # cm = confusion_matrix(results.T[0], results.T[1]).T
    path = f"{yolo_result_dir}/cm_{stage}_num.jpg"
    vis.plot_cm(cm, labels, path, False)

    # cm = confusion_matrix(results.T[0], results.T[1], normalize="true").T
    cm_recall = cm / (cm.sum(axis=0, keepdims=True) + 1e-9)
    path = f"{yolo_result_dir}/cm_{stage}_recall.jpg"
    vis.plot_cm(cm_recall, labels, path, True)

    # cm = confusion_matrix(results.T[0], results.T[1], normalize="pred").T
    cm_precision = cm / (cm.sum(axis=1, keepdims=True) + 1e-9)
    path = f"{yolo_result_dir}/cm_{stage}_precision.jpg"
    vis.plot_cm(cm_precision, labels, path, True)

    # cm = confusion_matrix(results.T[0], results.T[1], normalize="all").T
    cm_f1 = 2 / ((1 / (cm_recall + 1e-9)) + (1 / (cm_precision + 1e-9)) + 1e-9)
    path = f"{yolo_result_dir}/cm_{stage}_f1.jpg"
    vis.plot_cm(cm_f1, labels, path, True)

    path = f"{yolo_result_dir}/cm_{stage}_report.tsv"
    report = classification_report(
        results.T[0], results.T[1], digits=3, output_dict=True, zero_division=0
    )
    pd.DataFrame.from_dict(report).T.to_csv(path, sep="\t")
    return results, missed_img_paths
