import os
from glob import glob

from tqdm import tqdm


def collect_annotation_paint_images(ann_json, info_json):
    image_paths = sorted(glob(os.path.join("annotation/images", "*.jpg")))
    data = []
    for image_path in tqdm(image_paths, ncols=100):
        video_id, aid, _ = os.path.basename(image_path).split("_")
        if video_id not in info_json:
            # tqdm.write(f"{video_id} is not in info.json")
            continue

        ann_lst = [ann for ann in ann_json[video_id] if ann["reply"] == aid]
        if len(ann_lst) == 0:
            print("not found reply", video_id, aid)
            continue
        elif len(ann_lst) > 1:
            print("duplicated", video_id, aid, image_path)

        for i, ann in enumerate(ann_lst):
            label = ann["text"]

            try:
                label = label.split("(")[1].replace(")", "")  # extract within bracket
            except IndexError:
                print("error label", video_id, aid, label, image_path)
                continue

            # if i > 0:
            #     print("success", video_id, aid, label)
            break
        else:
            continue

        data.append((aid, label, os.path.basename(image_path)))

    return data


def collect_images_classification_dataset(
    video_name, th_sec, th_iou, bbox_ratio, img_dir="datasets/images"
):
    img_dir = f"{img_dir}/sec{th_sec}_iou{th_iou}_br{bbox_ratio}/{video_name}"
    anomaly_img_paths = sorted(glob(f"{img_dir}/1/**/*.jpg"))

    data = []
    for path in anomaly_img_paths:
        label = path.split("/")[-2]
        key = f"{video_name}-{label}"

        try:
            # extract within bracket
            label = label.split("(")[1].replace(")", "")
        except IndexError:
            print("error label", video_name, label)
            continue
        data.append((key, label, path))

    return data


def collect_images_anomaly_detection_dataset(
    video_name, th_sec, th_iou, bbox_ratio, img_dir="datasets/images"
):
    img_dir = f"{img_dir}/sec{th_sec}_iou{th_iou}_br{bbox_ratio}/{video_name}"
    anomaly_img_paths = sorted(glob(f"{img_dir}/1/**/*.jpg"))
    normal_img_paths = sorted(glob(f"{img_dir}/0/*.jpg"))

    data = []
    for img_path in anomaly_img_paths:
        key = f"{video_name}-1"
        data.append((key, 1, img_path))
    for img_path in normal_img_paths:
        key = f"{video_name}-0"
        data.append((key, 0, img_path))

    return data
