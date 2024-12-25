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
    roc_auc_score,
    roc_curve,
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
        for img in tqdm(imgs):
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            keypoints, descriptors = self._sift.detectAndCompute(gray, None)
            if descriptors is not None:
                self._trainer.add(descriptors)

    def create_vocablaries(self):
        voc = self._trainer.cluster()
        self._extractor.setVocabulary(voc)
        return voc

    def set_vocablaries(self, voc):
        self._extractor.setVocabulary(voc)

    def extract_keypoints(self, img):
        keypoints = self._sift.detect(img)
        # img_sift = cv2.drawKeypoints(img, keypoints, None, flags=4)
        # cv2.imwrite("sift_img.jpg", img_sift)
        return self._extractor.compute(img, keypoints)


def train_anomaly(
    data_name, split_type, img_size=(32, 32), n_vocabs=128, n_imgs=100000, seed=42
):
    data_root = f"datasets/anomaly/{data_name}/{split_type}"
    dataset_path = f"{data_root}/train.tsv"
    data = np.loadtxt(dataset_path, dtype=str, delimiter="\t")

    # select imgs
    print("loading train images")
    normal_img_paths = np.array([d[2] for d in data if int(d[1]) == 0])
    np.random.seed(seed)
    idxs = np.random.choice(len(normal_img_paths), n_imgs, replace=False)
    normal_img_paths = normal_img_paths[idxs]
    normal_imgs = [imread(path, img_size) for path in tqdm(normal_img_paths)]

    print("creating vocablaries")
    fe = FeatureExtractor(n_vocabs)
    fe.add_samples(normal_imgs)
    voc = fe.create_vocablaries()

    print("extracting keypoints")
    X = []
    for img in tqdm(normal_imgs):
        kps = fe.extract_keypoints(img)
        if kps is None:
            cv2.imwrite("none_img.jpg", img)
            continue
        X.append(kps)
    X = np.array(X)
    X = X.reshape(-1, n_vocabs)

    print("training")
    model = svm.OneClassSVM(nu=0.001, kernel="rbf", gamma=0.1)
    model.fit(X)

    result_dir = f"runs/anomaly/{data_name}/{split_type}"
    if os.path.exists(result_dir):
        dirs = sorted(glob(result_dir + "-v*/"))
        if len(dirs) == 0:
            v_num = 1
        else:
            last_dir = dirs[-1]
            v_num = int(os.path.dirname(last_dir).split("-")[-1].replace("v", "")) + 1
        result_dir = f"runs/anomaly/{data_name}/{split_type}-v{v_num}"

    os.makedirs(result_dir, exist_ok=True)
    voc_path = f"{result_dir}/vocabs.npy"
    np.save(voc_path, voc)
    model_path = f"{result_dir}/model.pkl"
    with open(model_path, "wb") as f:
        pickle.dump(model, f)

    return result_dir


def pred_anomaly(
    data_name, split_type, stage, result_dir, img_size=(32, 32), n_vocabs=128
):
    data_root = f"datasets/anomaly/{data_name}/{split_type}"
    dataset_path = f"{data_root}/{stage}.tsv"
    data = np.loadtxt(dataset_path, dtype=str, delimiter="\t")

    voc_path = f"{result_dir}/vocabs.npy"
    voc = np.load(voc_path)
    fe = FeatureExtractor(n_vocabs)
    fe.set_vocablaries(voc)

    model_path = f"{result_dir}/model.pkl"
    with open(model_path, "rb") as f:
        model = pickle.load(f)

    print("predicting", stage, "dataset")
    results = []
    for _, label, img_path in tqdm(data):
        img = imread(img_path, img_size)
        kps = fe.extract_keypoints(img)
        if kps is None:
            continue
        pred = model.predict(kps).item()
        pred = 0 if pred == -1 else 1
        results.append([int(label), pred])

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

    fpr, tpr, th = roc_curve(results.T[0], results.T[1])
    rocauc = roc_auc_score(results.T[0], results.T[1])
    plt.figure(figsize=(5, 5))
    plt.plot(fpr, tpr, "-o")
    plt.xlabel("FPR")
    plt.ylabel("TPR")
    plt.title(f"ROCAUC {rocauc:.3f}")
    path = f"{result_dir}/{stage}_rocauc.jpg"
    plt.savefig(path, bbox_inches="tight")
    plt.close()

    return results


def cm_plot(cm, path):
    cmd = ConfusionMatrixDisplay(cm)
    cmd.plot(xticks_rotation="vertical", include_values=True, cmap="Blues")
    plt.xticks(fontsize=6)
    plt.yticks(fontsize=6)
    plt.savefig(path, bbox_inches="tight")
    plt.close()
