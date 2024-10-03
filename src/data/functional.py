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

    return (x1, y1, x2, y2)
