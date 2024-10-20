import numpy as np


def calc_ious(target_bbox, bboxs):
    bboxs = np.asarray(bboxs)
    a_area = (target_bbox[2] - target_bbox[0]) * (target_bbox[3] - target_bbox[1])
    b_area = (bboxs[:, 2] - bboxs[:, 0]) * (bboxs[:, 3] - bboxs[:, 1])

    intersection_xmin = np.maximum(target_bbox[0], bboxs[:, 0])
    intersection_ymin = np.maximum(target_bbox[1], bboxs[:, 1])
    intersection_xmax = np.minimum(target_bbox[2], bboxs[:, 2])
    intersection_ymax = np.minimum(target_bbox[3], bboxs[:, 3])

    intersection_w = np.maximum(0, intersection_xmax - intersection_xmin)
    intersection_h = np.maximum(0, intersection_ymax - intersection_ymin)

    intersection_area = intersection_w * intersection_h
    union_area = a_area + b_area - intersection_area

    return intersection_area / union_area


def calc_resized_bbox(bbox, bbox_ratio, frame_size):
    x1, y1, x2, y2 = bbox
    xc = (x2 - x1) / 2 + x1
    yc = (y2 - y1) / 2 + y1
    length = np.linalg.norm(np.array(frame_size)) * bbox_ratio
    x1 = int(xc - length / 2)
    y1 = int(yc - length / 2)
    x2 = int(xc + length / 2)
    y2 = int(yc + length / 2)

    return np.array([x1, y1, x2, y2])


def split_train_test_by_video(data, video_id_to_name):
    # data ("{video_name}-{n(label)}, label, img")

    train_video_ids = np.loadtxt(
        "datasets/yolov8_finetuning/summary_dataset_train.tsv",
        dtype=str,
        delimiter="\t",
        skiprows=1,
    )[:-1, 0]
    test_video_ids = np.loadtxt(
        "datasets/yolov8_finetuning/summary_dataset_test.tsv",
        dtype=str,
        delimiter="\t",
        skiprows=1,
    )[:-1, 0]
    train_video_names = [video_id_to_name[_id] for _id in train_video_ids]
    test_video_names = [video_id_to_name[_id] for _id in test_video_ids]

    # split data per label
    train_idxs = []
    test_idxs = []
    for i, d in enumerate(data):
        video_name = d[0].split("-")[0]
        if video_name in train_video_names:
            train_idxs.append(i)
        elif video_name in test_video_names:
            test_idxs.append(i)

    return train_idxs, test_idxs


def split_train_test_by_annotation(data, seed=42):
    # data ("{video_name}-{n(label)}, label, img")
    np.random.seed(seed)

    idxs_dict = {}
    for i, d in enumerate(data):
        key = d[0]
        if key not in idxs_dict:
            idxs_dict[key] = []
        idxs_dict[key].append(i)

    unique_keys = list(idxs_dict.keys())
    random_keys = np.random.choice(unique_keys, len(unique_keys))
    train_length = int(len(unique_keys) * 0.7)
    train_keys = random_keys[:train_length]
    test_keys = random_keys[train_length:]

    train_idxs = []
    for key in train_keys:
        train_idxs += idxs_dict[key]
    test_idxs = []
    for key in test_keys:
        test_idxs += idxs_dict[key]

    return train_idxs, test_idxs
