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
    # img = cv2.resize(img, img_size)
    # img = (img / 255) * 2 - 1
    return img


class FeatureExtractor:
    def __init__(self, n_vocabularies):
        self._sift = cv2.SIFT_create()
        self._trainer = cv2.BOWKMeansTrainer(n_vocabularies)
        index_params = dict(algorithm=1, trees=5)
        search_params = {}
        flann = cv2.FlannBasedMatcher(index_params, search_params)
        self._extractor = cv2.BOWImgDescriptorExtractor(self._sift, flann)

    def add_samples(self, imgs):
        for img in imgs:
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            keypoints, descriptors = self._sift.detectAndCompute(gray, None)
            self._trainer.add(descriptors)

    def create_vocablaries(self):
        voc = self._trainer.cluster()
        self._extractor.setVocabulary(voc)
        return voc

    def extract_keypoints(self, img):
        keypoints = self._extractor.detect(img)
        img_sift = cv2.drawKeypoints(img, keypoints, None, flags=4)
        cv2.imwrite("sift_img.jpg", img_sift)
        raise KeyError
        return self._extractor.compute(img, keypoints)


def train_anomaly(data_name, split_type, img_size=(32, 32)):
    data_root = f"datasets/anomaly/{data_name}/{split_type}"
    dataset_path = f"{data_root}/train.tsv"
    data = np.loadtxt(dataset_path, dtype=str, delimiter="\t")[:1000]
    normal_img_paths = [d[2] for d in data if int(d[1]) == 0]
    normal_imgs = [imread(path, img_size) for path in tqdm(normal_img_paths)]

    fe = FeatureExtractor(128)
    fe.add_samples(normal_imgs)
    fe.create_vocablaries()
    fe.extract_keypoints(normal_imgs[0])

    X = np.array(normal_imgs)
    X = X.reshape(X.shape[0], -1)

    model = svm.OneClassSVM(nu=0.1, kernel="rbf", gamma=0.1)
    model.fit(X)

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


def pred_anomaly(data_name, split_type, stage, result_dir, img_size=(32, 32)):
    data_root = f"datasets/anomaly/{data_name}/{split_type}"
    dataset_path = f"{data_root}/{stage}.tsv"
    data = np.loadtxt(dataset_path, dtype=str, delimiter="\t")

    model_path = f"{result_dir}/model.pkl"
    with open(model_path, "rb") as f:
        model = pickle.load(f)

    results = []
    missed_img_paths = []
    for _, label, img_path in tqdm(data):
        img = imread(img_path, img_size)
        img = img.reshape(1, -1)
        pred = model.predict(img).item()
        pred = 0 if pred == -1 else 1
        results.append([label, pred])
        if label != pred:
            missed_img_paths.append([img_path, label, pred])

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
