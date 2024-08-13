import argparse
import sys
import warnings

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
from src.utils import json_handler, video

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
    video_name,
    cap,
    yolo_preds,
    inp,
    n_imgs,
    tokenizer,
    model_llava,
    image_processor,
    conv_mode,
    temperature=0.2,
    max_new_tokens=512,
):
    llava_preds = []
    with torch.inference_mode():
        for i, pred in enumerate(tqdm(yolo_preds, ncols=100, leave=False, desc=video_name)):
            x1, y1, x2, y2 = pred[1:5].astype(int)
            ret, frame = cap.read(pred[0])
            if not ret:
                ValueError
            img = frame[y1:y2, x1:x2]
            imgs = [img]

            # create input image tensor
            imgs_tensor = []
            for img in imgs:
                img = process_images(img, image_processor, {})
                img = img.to(model_llava.device, dtype=torch.float16)
                imgs_tensor.append(img)

            conv, input_ids, streamer, stopping_criteria = create_input_ids_streamer(
                inp, imgs, conv_mode, tokenizer
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

            conf = np.round(pred[5], 3)
            cls_id = pred[6]
            llava_preds.append([i, conf, int(cls_id), 1, outputs])
    return llava_preds


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-vn", "--video_name", type=str, required=False, default=None)
    parser.add_argument("-pv", "--prompt_version", type=int, required=False, default=0)
    parser.add_argument("-ni", "--n_images", type=int, required=False, default=1)

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

    video_name = args.video_name
    prompt_v = args.prompt_version
    n_imgs = args.n_images

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
    video_names = sorted([video_id_to_name[vid] for vid in info_json.keys()])
    if video_name is not None:
        assert video_name in video_names
        video_names = [video_name]

    for video_name in tqdm(video_names, ncols=100):
        # load video and yolo preds
        cap = video.Capture(f"video/{video_name}.mp4")
        frame_count = cap.frame_count
        frame_size = cap.size
        yolo_preds = np.loadtxt(
            f"out/{video_name}/{video_name}_det_finetuned.tsv", skiprows=1, dtype=float
        )

        llava_preds = pred_llava(
            video_name,
            cap,
            yolo_preds,
            inp,
            n_imgs,
            tokenizer,
            model_llava,
            image_processor,
            conv_mode,
            args.temperature,
            args.max_new_tokens,
        )

        cols = "tid\tyolo_conf\tyolo_cls\tn_imgs\tlabel"
        np.savetxt(
            f"out/{video_name}/{video_name}_llava_p{prompt_v}_each.tsv",
            llava_preds,
            fmt="%s",
            delimiter="\t",
            header=cols,
            comments="",
        )
