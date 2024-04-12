from ultralytics import YOLO

from transformers import CLIPProcessor, CLIPModel
import pickle
import json
import os
import glob
import torch
#########
import torch

from llava.constants import IMAGE_TOKEN_INDEX, DEFAULT_IMAGE_TOKEN, DEFAULT_IM_START_TOKEN, DEFAULT_IM_END_TOKEN
from llava.conversation import conv_templates, SeparatorStyle
from llava.model.builder import load_pretrained_model
from llava.utils import disable_torch_init
from llava.mm_utils import process_images, tokenizer_image_token, get_model_name_from_path, KeywordsStoppingCriteria

from PIL import Image

import requests
from PIL import Image
from io import BytesIO
from transformers import TextStreamer

mm_use_im_start_end=False
temperature=0.2
max_new_tokens=512

def process(image, tokenizer, model, image_processor, context_len, conv_mode):
    conv = conv_templates[conv_mode].copy()
    if "mpt" in conv_mode.lower():
        roles = ('user', 'assistant')
    else:
        roles = conv.roles
 
    # Similar operation in model_worker.py
    print(image.shape)
    image_tensor = process_images([image], image_processor, {})
    if type(image_tensor) is list:
        image_tensor = [image.to(model.device, dtype=torch.float16) for image in image_tensor]
    else:
        image_tensor = image_tensor.to(model.device, dtype=torch.float16)

    #inp = "What is this device which is used in a laboratory bench. Please respond with only one phrase."
    inp = "What is this device which is used in a laboratory bench. Please select one from the following list and respond with only one phrase: \
        '50 mL centrifuge tube with green lid', '15mL centrifuge tube with green lid', 'culture dish', 'pasteur pipette', 'blue 5ml serological pipette', 'orange 10ml serological pipette', 'green 10ml serological pipette', 'pipette filler', 'ventilation door',  and 'other'"
    print(f"{roles[0]}: "+inp)

    print(f"{roles[1]}: ", end="")

    if image is not None:
        # first message
        if mm_use_im_start_end:
            inp = DEFAULT_IM_START_TOKEN + DEFAULT_IMAGE_TOKEN + DEFAULT_IM_END_TOKEN + '\n' + inp
        else:
            inp = DEFAULT_IMAGE_TOKEN + '\n' + inp
        conv.append_message(conv.roles[0], inp)
        image = None
    else:
        # later messages
        conv.append_message(conv.roles[0], inp)
    conv.append_message(conv.roles[1], None)
    prompt = conv.get_prompt()

    input_ids = tokenizer_image_token(prompt, tokenizer, IMAGE_TOKEN_INDEX, return_tensors='pt').unsqueeze(0).cuda()
    stop_str = conv.sep if conv.sep_style != SeparatorStyle.TWO else conv.sep2
    keywords = [stop_str]
    stopping_criteria = KeywordsStoppingCriteria(keywords, tokenizer, input_ids)
    streamer = TextStreamer(tokenizer, skip_prompt=True, skip_special_tokens=True)

    with torch.inference_mode():
        output_ids = model.generate(
            input_ids,
            images=image_tensor,
            do_sample=True,
            temperature=temperature,
            max_new_tokens=max_new_tokens,
            streamer=streamer,
            use_cache=True,
            stopping_criteria=[stopping_criteria])

    outputs = tokenizer.decode(output_ids[0, input_ids.shape[1]:]).strip().replace("</s>","")
    conv.messages[-1][-1] = outputs
    print("\n", {"prompt": prompt, "outputs": outputs}, "\n")
    return outputs



def get_llava():
    # Model
    disable_torch_init()
    model_path="liuhaotian/llava-v1.5-13b"
    model_name = get_model_name_from_path(model_path)
    model_base=None
    load_8bit=False
    load_4bit=False
    device="cuda"
    tokenizer, model, image_processor, context_len = load_pretrained_model(model_path, model_base, model_name, load_8bit, load_4bit, device=device)

    if 'llama-2' in model_name.lower():
        conv_mode = "llava_llama_2"
    elif "v1" in model_name.lower():
        conv_mode = "llava_v1"
    elif "mpt" in model_name.lower():
        conv_mode = "mpt"
    else:
        conv_mode = "llava_v0"

    if conv_mode:
        print('[WARNING] the auto inferred conversation mode is {}'.format(conv_mode))

   
    return tokenizer, model, image_processor, context_len, conv_mode

#########

target_path="video20230627"
out_path="video20230627/out2"

os.makedirs(out_path,exist_ok=True)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

tokenizer, model_llava, image_processor, context_len, conv_mode = get_llava()
#clip_model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32")
#clip_processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")
#clip_model.to(device)
#transform = transforms.Compose([transforms.ToTensor()])

# load pretrained model.
model: YOLO = YOLO(model="yolov8x.pt")
#model: YOLO = YOLO(model="yolov8x-seg.pt")

def run(path, out_name="out"):
    #result = model.predict("sample_small.mp4", save=True, conf=0.1)
    result = model.predict(path, save=True, conf=0.1, save_txt=False)
    #print(result[0])
    
    ## save pickle(too large)
    #with open("output/"+out_name+".pkl","wb") as fp:
    #    pickle.dump(result, fp)

    #clip_label=["Test tube","yellow lid", "Flask", "Pipette", "Petri dish", "Tweezers"]
    with open("output/"+out_name+".tsv","w") as fp:
        arr=["frame","id","x","y","w","h","class","class_name","confidence", "llava_label"]
        s="\t".join(map(str,arr))
        fp.write(s)
        fp.write("\n")
        for i,r in enumerate(result):
            if i%10!=0:
                continue
            n=len(r.boxes.cls)
            for j in range(n):
                xywh=r.boxes.xywh[j].cpu().numpy()
                x,y,w,h = int(xywh[0]), int(xywh[1]),int(xywh[2]),int(xywh[3])
                if w*h>100 and w*h<384*640/4:
                    w+=100
                    h+=100
                    #img = r.orig_img[y:y+h, x:x+w]
                    yy=y-h//2
                    xx=x-w//2
                    if xx<0:
                        xx=0
                    if yy<0:
                        yy=0
                    img = r.orig_img[yy:y+h//2, xx:x+w//2]
                    if img.shape[0]*img.shape[1]<100:
                        continue
                    #print("?1>>", r.orig_img.shape)
                    #print("?2>>",img.shape)
                    #print("?3>>",x,y,w,h)

                    #inputs = clip_processor(text=clip_label, images=img, return_tensors="pt", padding=True)
                    #inputs.to(device)
                    #outputs = clip_model(**inputs)

                    #logits_per_image = outputs.logits_per_image # this is the image-text similarity score
                    #probs = logits_per_image.softmax(dim=1) # we can take the softmax to get the label probabilities
                    #probs = probs.cpu().detach().numpy().tolist()[0]
                    out=process(img, tokenizer, model_llava, image_processor, context_len, conv_mode)
                    if False:
                        if i%100==0:
                            pil_img = Image.fromarray(img)
                            pil_img.save("temp/{}_{}_{}.jpg".format(i,j,out))
                    #print("?>>",probs)
                    
                    
                    cls=r.boxes.cls[j].item()
                    name=r.names[cls]
                    conf=r.boxes.conf[j].item()
                    
                    arr=[i,j,xywh[0],xywh[1],xywh[2],xywh[3],cls,name,conf]+[out]
                    s="\t".join(map(str,arr))
                    fp.write(s)
                    fp.write("\n")

#run("sample_small.mp4")
import os
for filename in glob.glob("video/*.mp4"):
    name,_=os.path.splitext(os.path.basename(filename))
    print(name)
    run(filename, name)
quit()
# <figure class="figure-image figure-image-fotolife" title="YOLOv8xの推論結果のgif">[f:id:yasaka_uta:20230327173819g:plain]<figcaption>YOLOv8xの推論結果のgif</figcaption></figure>inference
# save flgをTrueにすることで推論結果を描画した動画が保存される。
#result = model.predict("sample_small.mp4", save=True, conf=0.1)
for filename in glob.glob(target_path+"/*.json"):
    arr=filename.split("-")
    e=arr[1]
    path=target_path+"/original/{}.mp4".format(e)
    if os.path.isfile(path):
        print(path)
        print(filename)
        run(path)

