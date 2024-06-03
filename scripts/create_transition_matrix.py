import os
import re
import sys

import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import ConfusionMatrixDisplay, confusion_matrix

sys.path.append("src")
from utils import json_handler

ann_json_path = "annotation/annotation.json"
ann_json = json_handler.load(ann_json_path)
info_json = json_handler.load("annotation/info.json")

pattern = "(?<=\().+?(?=\))"
# pattern = "\(.*?\)"

transitions_all = []
for video_id, ann_data in ann_json.items():
    if video_id not in info_json:
        print(f"{video_id} is not in info.json")
        continue

    transitions = [(-1, "START")]
    for ann in ann_data:
        if ann["type"] == "TextComment":
            if "終了" not in ann["text"]:
                count = int(ann["text"].split("(")[0])
                label = re.findall(pattern, ann["text"])
                if len(label) > 0:
                    label = f"{label[0]}_S"
                    transitions.append((count, label))
                else:
                    transitions.append((None, "ERROR"))
            else:
                count_str = ann["text"].replace("終了", "")
                for count in count_str.split(","):
                    count = int(count)
                    label = [data[1] for data in transitions if data[0] == count][
                        0
                    ].split("_")[0]
                    label = f"{label}_E"
                    transitions.append((count, label))
    else:
        transitions.append((-1, "END"))

    pre_data = transitions[0]
    for data in transitions[1:]:
        transitions_all.append([pre_data[1], data[1]])
        pre_data = data

transitions_all = np.array(transitions_all).T

# sort labels
labels = np.unique(transitions_all).tolist()
labels.remove("START")
labels.remove("END")
if "ERROR" in labels:
    labels.remove("ERROR")
labels = sorted(labels, key=lambda x: x.split("_")[1], reverse=True)
labels = sorted(labels, key=lambda x: x.split("_")[0])
labels = ["START"] + labels + ["ERROR", "END"]

cm = confusion_matrix(transitions_all[0], transitions_all[1], labels=labels)
cmd = ConfusionMatrixDisplay(cm, display_labels=labels)
cmd.plot(xticks_rotation="vertical", include_values=False)
plt.xticks(fontsize=6)
plt.yticks(fontsize=6)
plt.xlabel("next state")
plt.ylabel("pre state")
path = "out/transition_matrix/transition_matrix.jpg"
os.makedirs(os.path.dirname(path), exist_ok=True)
plt.savefig(path, bbox_inches="tight")
plt.show()

clip_max = 10
cm = confusion_matrix(transitions_all[0], transitions_all[1], labels=labels)
cm = np.clip(cm, 0, clip_max)
cmd = ConfusionMatrixDisplay(cm, display_labels=labels)
cmd.plot(xticks_rotation="vertical", include_values=False)
plt.xticks(fontsize=6)
plt.yticks(fontsize=6)
plt.xlabel("next state")
plt.ylabel("pre state")
plt.savefig(
    f"out/transition_matrix/transition_matrix_clip{clip_max}.jpg", bbox_inches="tight"
)
plt.show()
