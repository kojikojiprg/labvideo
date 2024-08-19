import argparse
import os
import sys
import warnings
from glob import glob

import cv2
import numpy as np
import torch
from llava.constants import DEFAULT_IMAGE_TOKEN, IMAGE_TOKEN_INDEX
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
from src.utils import json_handler

warnings.filterwarnings("ignore")


def create_prompt(prompt_path, categories_path):
    with open(prompt_path, "r") as f:
        prompt = f.read()

    with open(categories_path, "r") as f:
        categories_lst = f.readlines()
    categories_str = "'"
    for i, c in enumerate(categories_lst):
        if i < len(categories_lst) - 1:
            categories_str += c.replace("\n", "','")
        else:
            categories_str += c.replace("\n", "'")

    prompt = prompt.replace("<categories>", categories_str)
    return prompt


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


def create_input_ids_streamer(inp, imgs, conv_mode, tokenizer):
    conv = conv_templates[conv_mode].copy()
    for _ in range(len(imgs)):
        inp = DEFAULT_IMAGE_TOKEN + inp

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

    return conv, input_ids, streamer, stopping_criteria


def pred_llava(
    image_paths,
    label_paths,
    inp,
    tokenizer,
    model_llava,
    image_processor,
    conv_mode,
    temperature=0.2,
    max_new_tokens=512,
):
    llava_preds = []
    with torch.inference_mode():
        for i, (img_path, label_path) in enumerate(
            zip(tqdm(image_paths, ncols=100, leave=False), label_paths)
        ):
            frame = cv2.imread(img_path)
            labels = np.loadtxt(label_path, delimiter=" ").astype(np.float32)

            fh, fw = frame.shape[:2]
            for label in labels:
                x, y, w, h = label[1:]
                x1 = int((x - (w / 2)) * fw)
                y1 = int((y - (h / 2)) * fh)
                x2 = int((x + (w / 2)) * fw)
                y2 = int((y + (h / 2)) * fh)
                img = frame[y1:y2, x1:x2]
                imgs = [img]

                # create input image tensor
                imgs_tensor = []
                for img in imgs:
                    img = process_images(img, image_processor, {})
                    img = img.to(model_llava.device, dtype=torch.float16)
                    imgs_tensor.append(img)

                conv, input_ids, streamer, stopping_criteria = (
                    create_input_ids_streamer(inp, imgs, conv_mode, tokenizer)
                )

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

                outputs = (
                    tokenizer.decode(output_ids[0]).strip().replace("</s>", "")
                ).replace("<s>", "")
                conv.messages[-1][-1] = outputs

                outputs = outputs.strip().lower()
                pred_id = output2clsid[outputs]
                true_id = int(label[0])
                llava_preds.append(
                    [i, os.path.basename(img_path), outputs, pred_id, true_id]
                )
    return llava_preds


output2clsid = {
    "centrifuge tube": 0,
    "centrifuge tube cap": 1,
    "culture dish": 2,
    "pipette": 3,
    "dispenser": 4,
    "hand": 5,
}


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-pv", "--prompt_version", type=int, required=False, default=0)

    parser.add_argument(
        "-mp",
        "--model_path",
        type=str,
        required=False,
        default="liuhaotian/llava-v1.5-13b",
    )
    parser.add_argument("-mb", "--model_base", type=str, required=False, default=None)
    parser.add_argument(
        "-l8", "--load_8bit", required=False, default=False, action="store_true"
    )
    parser.add_argument(
        "-l4", "--load_4bit", required=False, default=False, action="store_true"
    )
    parser.add_argument("-tm", "--temperature", type=float, required=False, default=0.2)
    parser.add_argument(
        "-mnt", "--max_new_tokens", type=int, required=False, default=512
    )
    args = parser.parse_args()

    prompt_v = args.prompt_version

    # create prompt
    prompt_path = f"prompts/prompt_v{prompt_v}.txt"
    categories_path = "prompts/categories.txt"
    inp = create_prompt(prompt_path, categories_path)

    # load llava
    tokenizer, model_llava, image_processor, context_len, conv_mode = get_llava(
        args.model_path, args.model_base, args.load_8bit, args.load_4bit
    )

    info_json = json_handler.load("annotation/info.json")
    video_id_to_name = {
        data[0]: data[1].split(".")[0]
        for data in np.loadtxt(
            "annotation/annotation.tsv",
            str,
            delimiter="\t",
            skiprows=1,
            usecols=[1, 2],
        )
        if data[0] != "" and data[1] != ""
    }
    train_image_paths = sorted(glob("datasets/yolov8_finetuning/images/train/*.jpg"))
    test_image_paths = sorted(glob("datasets/yolov8_finetuning/images/test/*.jpg"))
    train_label_paths = sorted(glob("datasets/yolov8_finetuning/labels/train/*.txt"))
    test_label_paths = sorted(glob("datasets/yolov8_finetuning/labels/test/*.txt"))

    cols = "\timg_name\tpred\tpred_id\ttrue_id"

    # train dataset
    llava_preds = pred_llava(
        train_image_paths,
        train_label_paths,
        inp,
        tokenizer,
        model_llava,
        image_processor,
        conv_mode,
        args.temperature,
        args.max_new_tokens,
    )
    np.savetxt(
        f"out/llava_p{prompt_v}_finetuning_train.tsv",
        llava_preds,
        fmt="%s",
        delimiter="\t",
        header=cols,
        comments="",
    )

    # test dataset
    llava_preds = pred_llava(
        test_image_paths,
        test_label_paths,
        inp,
        tokenizer,
        model_llava,
        image_processor,
        conv_mode,
        args.temperature,
        args.max_new_tokens,
    )
    np.savetxt(
        f"out/llava_p{prompt_v}_finetuning_test.tsv",
        llava_preds,
        fmt="%s",
        delimiter="\t",
        header=cols,
        comments="",
    )
