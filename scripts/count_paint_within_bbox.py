import os
from glob import glob

import numpy as np
from tqdm import tqdm

dirs = sorted(glob("out/*/"))
dirs = [dir_path for dir_path in dirs if len(glob(f"{dir_path}/*_ann.tsv")) > 0]

results = []
for dir_path in tqdm(dirs, ncols=100):
    video_name = os.path.basename(os.path.dirname(dir_path))

    ann_data = np.loadtxt(f"{dir_path}/{video_name}_ann.tsv", skiprows=1, dtype=str)
    yolo_preds = np.loadtxt(f"{dir_path}/{video_name}_det.tsv", skiprows=1, dtype=float)
    if len(yolo_preds) == 0:
        continue

    count = 0
    max_n_frame = int(max([max(yolo_preds.T[0]), max(ann_data.T[0].astype(int))]))
    for n_frame in tqdm(range(max_n_frame), ncols=100, leave=False):
        ann_tmp = [ann for ann in ann_data if int(ann[0]) == n_frame]
        preds = [p for p in yolo_preds if int(p[0]) == n_frame]
        if len(preds) == 0:
            continue
        for ann in ann_tmp:
            xc, yc = ann[5:7].astype(np.float32)
            for pred in preds:
                x1, y1, x2, y2 = pred[1:5]
                is_in_x = np.any(x1 <= xc) & np.any(xc <= x2)
                is_in_y = np.any(y1 <= yc) & np.any(yc <= y2)
                if is_in_x and is_in_y:
                    count += 1

    results.append(
        (video_name, f"{len(ann_data)}", f"{count}", f"{count / len(ann_data):.3f}")
    )

header = "video_name\ttotal\tcount\tratio"
np.savetxt("out/count_patin_within_bbox.tsv", results, "%s", header=header, comments="")
