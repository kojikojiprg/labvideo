import os
import sys
import warnings

import cv2
import numpy as np
import torch
from llava.constants import (
    DEFAULT_IM_END_TOKEN,
    DEFAULT_IM_START_TOKEN,
    DEFAULT_IMAGE_TOKEN,
    IMAGE_TOKEN_INDEX,
)
from llava.conversation import SeparatorStyle, conv_templates
from llava.mm_utils import (
    KeywordsStoppingCriteria,
    get_model_name_from_path,
    process_images,
    tokenizer_image_token,
)
from llava.model.builder import load_pretrained_model
from llava.utils import disable_torch_init
from tqdm import tqdm
from transformers import TextStreamer

sys.path.append(".")
from src.utils import video

warnings.filterwarnings("ignore")

# load yolo data
video_name = "Failures11"
# video_name = "継代12"
yolo_preds = np.loadtxt(
    f"out/{video_name}/{video_name}_det.tsv", skiprows=1, dtype=float
)

cap = video.Capture(f"video/{video_name}.mp4")
frame_count = cap.frame_count
frame_size = cap.size


# params
temperature = 0.2
max_new_tokens = 512

# llava params
model_path = "liuhaotian/llava-v1.5-13b"
model_base = None
load_8bit = False
load_4bit = False

# load prompt
prompt_v = 0
prompt_path = f"prompts/prompt_v{prompt_v}.txt"
with open(prompt_path, "r") as f:
    inp = f.read()


# load llava
def get_llava(model_path, model_base, load_8bit, load_4bit):
    # Model
    disable_torch_init()
    model_name = get_model_name_from_path(model_path)
    tokenizer, model, image_processor, context_len = load_pretrained_model(
        model_path, model_base, model_name, load_8bit, load_4bit, device="cuda"
    )

    if "llama-2" in model_name.lower():
        conv_mode = "llava_llama_2"
    elif "v1" in model_name.lower():
        conv_mode = "llava_v1"
    elif "mpt" in model_name.lower():
        conv_mode = "mpt"
    else:
        conv_mode = "llava_v0"

    # if conv_mode:
    #     print("[WARNING] the auto inferred conversation mode is {}".format(conv_mode))

    return tokenizer, model, image_processor, context_len, conv_mode


tokenizer, model_llava, image_processor, context_len, conv_mode = get_llava(
    model_path, model_base, load_8bit, load_4bit
)

# create prompt
conv = conv_templates[conv_mode].copy()
inp = DEFAULT_IMAGE_TOKEN + "\n" + inp

conv.append_message(conv.roles[0], inp)
conv.append_message(conv.roles[1], None)

prompt = conv.get_prompt()

# create streamer
input_ids = (
    tokenizer_image_token(prompt, tokenizer, IMAGE_TOKEN_INDEX, return_tensors="pt")
    .unsqueeze(0)
    .cuda()
)
stop_str = conv.sep if conv.sep_style != SeparatorStyle.TWO else conv.sep2
keywords = [stop_str]
stopping_criteria = KeywordsStoppingCriteria(keywords, tokenizer, input_ids)
streamer = TextStreamer(tokenizer, skip_prompt=True, skip_special_tokens=True)

# pred llava
n_imgs = 10

llava_preds = []
track_ids = np.unique(yolo_preds[:, 6]).astype(int)
with torch.inference_mode():
    for tid in tqdm(track_ids, ncols=100):
        preds_tmp = np.array([pred for pred in yolo_preds if pred[6] == tid])
        if len(preds_tmp) > n_imgs:
            pred_idxs = np.argpartition(-preds_tmp[:, 5], n_imgs)[:n_imgs]
            preds_tmp = preds_tmp[pred_idxs]

        img_dir = f"out/{video_name}/images"
        imgs = []
        for pred in preds_tmp:
            x1, y1, x2, y2 = pred[1:5].astype(int)
            ret, frame = cap.read(pred[0])
            if not ret:
                ValueError
            img = frame[y1:y2, x1:x2]
            imgs.append(img)
        cv2.imwrite(f"{img_dir}/tid{tid}.jpg", cv2.cvtColor(imgs[0], cv2.COLOR_RGB2BGR))

        imgs_tensor = process_images(imgs, image_processor, {})
        imgs_tensor = imgs_tensor.to(model_llava.device, dtype=torch.float16)

        output_ids = model_llava.generate(
            input_ids,
            images=imgs_tensor,
            do_sample=True,
            temperature=temperature,
            max_new_tokens=max_new_tokens,
            streamer=streamer,
            use_cache=True,
            stopping_criteria=[stopping_criteria],
        )

        outputs = (tokenizer.decode(output_ids[0]).strip().replace("</s>", "")).replace(
            "<s>", ""
        )
        conv.messages[-1][-1] = outputs

        # print("\nprompt")
        # print(prompt)
        # print("\nn_imgs", len(imgs))
        # print("\noutputs_ids")
        # print(output_ids.cpu().numpy())
        # print("\noutputs")
        # print(outputs)

        # if "," in outputs:
        #     label, conf = outputs.split(",")
        #     label = label.lower()
        #     conf = float(conf.replace(" ", ""))
        # elif "(" in outputs:
        #     label, conf = outputs.split("(")
        #     label = label.lower()
        #     conf = float(conf.replace(")", ""))
        # elif "with" in outputs:
        #     label, conf = outputs.split("with")
        #     lavel = label.replace("'", "")
        #     conf = float(conf[-5:-1])
        # else:
        #     ValueError
        # tqdm.write(f"{tid}, {label}, {conf}")
        # llava_preds.append([tid, label, conf])
        llava_preds.append([tid, outputs])

# cols = "tid\tlabel\tconf"
cols = "tid\tlabel"
np.savetxt(
    f"out/{video_name}/{video_name}_llava.tsv",
    llava_preds,
    fmt="%s",
    delimiter="\t",
    header=cols,
    comments="",
)
