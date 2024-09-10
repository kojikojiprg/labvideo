import os
import pickle
from glob import glob

import cv2
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn import svm
from sklearn.metrics import (
    ConfusionMatrixDisplay,
    classification_report,
    confusion_matrix,
)
from tqdm import tqdm


def imread(img_path, img_size):
    img = cv2.imread(img_path)
    img = cv2.resize(img, img_size)
    img = (img / 255) * 2 - 1
    return img

def create_dataset_anomaly(data, idxs, data_root, stage):
    for i, idx in enumerate(tqdm(idxs, ncols=100)):
        key, label, img_path = data[idx]
        img = cv2.imread(img_path)


def train_anomaly(data_name, split_type, img_size=(32, 32)):
    model = svm.OneClassSVM(nu=0.1, kernel="rbf", gamma=0.1)

    data_root = f"datasets/anomaly/{split_type}/{data_name}"
    img_paths = sorted(glob(f"{data_root}/train/0/*.jpg"))
    normal_imgs = [imread(path, img_size) for path in img_paths]
    model.fit(np.array(normal_imgs))

    result_dir = f"runs/anomaly/{split_type}/{data_name}"
    if os.path.exists(result_dir):
        dirs = sorted(glob(result_dir + "-v*/"))
        if len(dirs) == 0:
            v_num = 1
        else:
            last_dir = dirs[-1]
            v_num = int(os.path.dirname(last_dir).split("-")[-1].replace("v", "")) + 1
        result_dir = f"runs/classify/{data_name}-v{v_num}"

    os.makedirs(result_dir, exist_ok=True)
    model_path = f"{result_dir}/model.pkl"
    with open(model_path, "wb") as f:
        pickle.dump(model, f)

    return result_dir


def pred_anomaly(img_paths, stage, result_dir):
    model_path = f"{result_dir}/model.pkl"
    with open(model_path, "rb") as f:
        model = pickle.load(f)

    results = []
    missed_img_paths = []
    for path in tqdm(img_paths):
        label = os.path.basename(os.path.dirname(path))
        pred = model.predict(path)
        pred_label_id = pred[0].probs.top1
        names = pred[0].names
        pred_label = names[pred_label_id]
        results.append([label, pred_label])
        if label != pred_label:
            missed_img_paths.append([path, label, pred_label])

    results = np.array(results)
    cm = confusion_matrix(results.T[0], results.T[1])
    path = f"{result_dir}/cm_{stage}_num.jpg"
    cm_plot(cm, path)
    cm = confusion_matrix(results.T[0], results.T[1], normalize="true")
    path = f"{result_dir}/cm_{stage}_recall.jpg"
    cm_plot(cm, path)
    cm = confusion_matrix(results.T[0], results.T[1], normalize="pred")
    path = f"{result_dir}/cm_{stage}_precision.jpg"
    cm_plot(cm, path)
    cm = confusion_matrix(results.T[0], results.T[1], normalize="all")
    path = f"{result_dir}/cm_{stage}_f1.jpg"
    cm_plot(cm, path)

    path = f"{result_dir}/cm_{stage}_report.tsv"
    report = classification_report(
        results.T[0], results.T[1], digits=3, output_dict=True, zero_division=0
    )
    pd.DataFrame.from_dict(report).T.to_csv(path, sep="\t")
    return results, missed_img_paths


def cm_plot(cm, path):
    cmd = ConfusionMatrixDisplay(cm)
    cmd.plot(xticks_rotation="vertical", include_values=True, cmap="Blues")
    plt.xticks(fontsize=6)
    plt.yticks(fontsize=6)
    plt.savefig(path, bbox_inches="tight")
    plt.close()
