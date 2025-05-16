import torch
import PIL
import cv2
import os
import numpy as np
import mediapipe as mp
from utils.masking_utils import generate_mask_from_landmarks_simple,generate_mask_from_landmarks
from utils.mp_function_utils import find_landmarks
from utils.load_and_preprocessing_utils import load_img_pil
from utils.misc_utils import save_json,crop_and_make_square,clear_memory,make_folders_multi,display_multi


## memory reduction
## https://huggingface.co/docs/diffusers/en/optimization/memory

## -----------------------------------------------------
## For longer prompts i.e > 77
## -----------------------------------------------------

from sd_embed.embedding_funcs import get_weighted_text_embeddings_sdxl 
from utils.lpw_stable_diffusion_xl import get_weighted_text_embeddings_sdxl_lpw

def get_prompt_embeddings(pipe,prompt,negative_prompt,embed_type):
    '''To solve "max token=77" problem'''

    allowed_modes = ['sd_emb', 'lpw']
    assert embed_type in allowed_modes, f"Invalid embedding type selected, select from: {allowed_modes}"
    if embed_type=='sd_emb':
        embeded_prompts = get_weighted_text_embeddings_sdxl(pipe,prompt=prompt, neg_prompt=negative_prompt)
    elif embed_type=='lpw':
        embeded_prompts = get_weighted_text_embeddings_sdxl_lpw(pipe,prompt,negative_prompt)

    return embeded_prompts

from diffusers import (
    StableDiffusionPipeline,
    StableDiffusionInpaintPipeline,
    StableDiffusionXLPipeline,
    StableDiffusionXLInpaintPipeline,
    DPMSolverMultistepScheduler,
)

## -----------------------------------------------------
## loading models
## -----------------------------------------------------

def load_diffusion_model(
    model_path: str,
    model_type: str = "standard",  # one of: "standard", "inpainting", "xl", "xl-inpainting"
    torch_dtype: torch.dtype = None,
    use_safetensors: bool = True,
    custom_scheduler=None,
    **kwargs
):
    """
    Load a Stable Diffusion model with flexible options.
    
    Args:
        model_path (str): Path or model ID (from Hugging Face hub or local folder)
        model_type (str): Type of model - "standard", "inpainting", "xl", "xl-inpainting"
        torch_dtype (torch.dtype): torch.float16 or torch.float32. Auto if None.
        use_safetensors (bool): Whether to load from .safetensors if available
        custom_scheduler: Custom scheduler instance (optional)
        device (str): Device to load on ("cuda" or "cpu")
        **kwargs: Additional arguments to pass to pipeline

    Returns:
        Diffusers pipeline object
    """

    torch_dtype = torch_dtype or (torch.float16 if torch.cuda.is_available() else torch.float32)

    pipe_cls_map = {
        "standard": StableDiffusionPipeline,
        "inpainting": StableDiffusionInpaintPipeline,
        "xl": StableDiffusionXLPipeline,
        "xl-inpainting": StableDiffusionXLInpaintPipeline,
    }

    if model_type not in pipe_cls_map:
        raise ValueError(f"Invalid model_type: {model_type}")

    pipeline_cls = pipe_cls_map[model_type]

    # Load pipeline
    if '.safetensors' in model_path:
        pipe = pipeline_cls.from_single_file(
            model_path,
            torch_dtype=torch_dtype,
            use_safetensors=use_safetensors,
            **kwargs
        )
    else:    
        pipe = pipeline_cls.from_pretrained(
            model_path,
            torch_dtype=torch_dtype,
            use_safetensors=use_safetensors,
            **kwargs
        )

    # Optional: Replace scheduler
    if custom_scheduler is not None:
        pipe.scheduler = custom_scheduler.from_config(pipe.scheduler.config)

    return pipe

## -----------------------------------------------------
## Lora related
## -----------------------------------------------------

def load_loras_into_pipeline(pipe, lora_sources):
    """
    Load multiple LoRAs into an SDXL pipeline with named adapters and apply them with weights.

    Args:
        pipe (StableDiffusionXLPipeline): The Diffusers pipeline.
        lora_sources (list of dict): Each dict has:
            - 'path': str, local file or HF repo_id
            - 'weight_name': str (optional, required for HF)
            - 'weight': float, influence (default = 1.0)
            - 'name': str, unique adapter name
    """
    adapter_names = []
    adapter_weights = []

    for lora in lora_sources:
        name = lora["name"]
        path = lora["path"]
        weight = lora.get("weight", 1.0)
        weight_name = lora.get("weight_name", None)

        if os.path.isfile(path):
            print(f"🔹 Loading local LoRA '{name}' from {path}")
            pipe.load_lora_weights(path, adapter_name=name)
        else:
            print(f"🔹 Loading HF LoRA '{name}' from {path} ({weight_name})")
            pipe.load_lora_weights(path, weight_name=weight_name, adapter_name=name)

        adapter_names.append(name)
        adapter_weights.append(weight)

    print(f"✅ Setting adapters: {adapter_names} with weights: {adapter_weights}")
    pipe.set_adapters(adapter_names, adapter_weights)

    return pipe


## -----------------------------------------------------
## misc
## -----------------------------------------------------

def load_and_generate_mask(FA_model,detector,img_path,masked_parts_list,dilate_list,img_shape=(1024,1024),face_front=False):
    img = load_img_pil(img_path)
    mp_img = mp.Image(image_format=mp.ImageFormat.SRGB, data=np.asarray(img))
    detection_result = detector.detect(mp_img)
    mp_img = np.copy(mp_img.numpy_view())
    padding_color = tuple(map(int, mp_img[0,0])) 
    img = crop_and_make_square(mp_img, detection_result,bbox_expand_ratio=1.5,padding_color=padding_color)
    img = PIL.Image.fromarray(img)
    img = img.resize(img_shape)
    
    # For face Front
    if face_front==True:
        img_landmarks = find_landmarks(np.array(img))
        img_mask,_ = generate_mask_from_landmarks(img,img_landmarks,masked_parts_list,dilate_list)
        img_mask = cv2.blur(img_mask,(21,21))
        # img_mask = 255*cv2.merge((img_mask,img_mask,img_mask))

    ## For side on face
    else:
        img_landmarks = FA_model.get_landmarks(np.array(img))[-1]
        img_mask,_ = generate_mask_from_landmarks_simple(img,img_landmarks,masked_parts_list,dilate_list)
        img_mask = cv2.blur(img_mask,(21,21))
        # landmark_img = draw_landmarks(np.array(img), img_landmarks, selected_indices=landmark_nose,scale=2.0, 
        #             color_pt=(0, 255, 0),color_txt=(255,0,0),
        #                 write_idx=True)

    mask_3d = 255*cv2.merge((img_mask,img_mask,img_mask))
    masked_img = np.array(img)*0.7 + np.array(mask_3d)*0.3

    return img,img_mask,masked_img


def perform_inpainting(pipe,FA_model,detector,all_paths,path_results,masked_parts_list,dilate_list,seed,prompt,negative_prompt,
                       guidance_scale=7.5,strength=1.0,lora_scale=1.0,num_steps=30,num_imgs=1,face_front=False,img_shape=(1024,1024),save_quality=100):
    print("inpainting")
    make_folders_multi(path_results)
    for img_path in all_paths:
        img,img_mask,masked_img = load_and_generate_mask(FA_model,detector,img_path,masked_parts_list,dilate_list,img_shape=img_shape,face_front=face_front)

        clear_memory()
        sd_images = pipe(prompt=[prompt]*num_imgs, image=img, negative_prompt=[negative_prompt]*num_imgs,
                mask_image=img_mask,
                height=img_shape[0],width=img_shape[1],
                num_inference_steps=num_steps,
                generator=torch.manual_seed(seed),
                cross_attention_kwargs={"scale":lora_scale},
                guidance_scale=guidance_scale,
                strength=strength
                ).images
        clear_memory()

        img_name,img_ext = os.path.splitext(img_path.split('/')[-1])
        for idx in range(num_imgs):
            # sd_img_name = os.path.join(path_results,f'{img_name}_SD{idx}{img_ext}')
            # cv2.imwrite(sd_img_name,np.array(sd_images[idx])[:,:,::-1],[int(cv2.IMWRITE_JPEG_QUALITY), save_quality])
            concat_img = np.concatenate((img,np.array(sd_images[idx]),masked_img),axis=1)
            concat_img_name = os.path.join(path_results,f'{img_name}_concat{idx}{img_ext}')
            cv2.imwrite(concat_img_name,concat_img[:,:,::-1],[int(cv2.IMWRITE_JPEG_QUALITY), save_quality])

    out_json = {'prompt':prompt,
                'negative_prompt':negative_prompt,
                'num_steps':num_steps,
                'guidance_scale':guidance_scale,
                'strength':strength,
                'seed':seed}
    
    save_json(img_name,out_json,path_results)

    return concat_img

def perform_diffusion(pipe,seed,prompt,negative_prompt,
                       guidance_scale=7.5,strength=1.0,lora_scale=1.0,num_steps=30,num_imgs=1,height=1024,width=1024):
    print("diffusion")
    
    clear_memory()
    sd_images = pipe(prompt=prompt, negative_prompt=negative_prompt,
            height=height,width=width,
            num_inference_steps=num_steps,
            generator=torch.manual_seed(seed),
            cross_attention_kwargs={"scale":lora_scale},
            guidance_scale=guidance_scale,
            strength=strength,
            num_images_per_prompt=num_imgs
            ).images
    clear_memory()

    return sd_images