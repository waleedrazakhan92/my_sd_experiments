import torch
from utils.load_and_preprocessing_utils import postprocess_image
from utils.masking_utils import replace_parts
import matplotlib.pyplot as plt
import numpy as np
import os
import cv2
import shutil
from PIL import Image
import io
import base64
from IPython.display import HTML, display
import json
from datetime import datetime
from PIL import ImageEnhance
from glob import glob
import time
import gc

def clear_memory():
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.synchronize() 
        torch.cuda.ipc_collect() 

def timer(func):
    def wrapper(*args, **kwargs):
        start = time.time()
        result = func(*args, **kwargs)
        end = time.time()
        print(f"Function '{func.__name__}' executed in {end - start:.4f} seconds")
        return result
    return wrapper


def add_gloss(image,gloss=1):
    image = Image.fromarray(np.array(image,dtype=np.uint8))
    enhancer = ImageEnhance.Contrast(image)
    image_glossy = enhancer.enhance(gloss)
    return np.array(image_glossy)


def read_paths(in_path: str, supported_extensions=None, recursive=False) -> list:
    """
    Reads and returns a list of image file paths from the given directory.

    Args:
        in_path (str): Path to the directory to scan for image files.
        supported_extensions (list, optional): List of supported image file extensions. Defaults to common image formats.
        recursive (bool, optional): Whether to include subdirectories. Defaults to False.

    Returns:
        list: List of image file paths.
    """
    if supported_extensions is None:
        supported_extensions = [".jpg", ".jpeg", ".png", ".bmp", ".tiff", ".webp"]

    image_paths = []

    if recursive:
        # Walk through subdirectories
        for root, _, files in os.walk(in_path):
            for file in files:
                if os.path.splitext(file)[1].lower() in supported_extensions:
                    image_paths.append(os.path.join(root, file))
    else:
        # Only scan the given directory (no subdirectories)
        image_paths = [
            os.path.join(in_path, file) 
            for file in os.listdir(in_path) 
            if os.path.isfile(os.path.join(in_path, file)) and os.path.splitext(file)[1].lower() in supported_extensions
        ]

    tot_files = len(glob(os.path.join(in_path, '*')))
    print('*' * 50)
    print(f"Total Files: {len(image_paths)}/{tot_files}")
    print('*' * 50)

    return image_paths


def get_date_time():
    # current date and time
    now = datetime.now()

    # tt = now.strftime("%H:%M:%S")
    # print("time:", tt)

    dt = now.strftime("%m-%d-%Y, %H:%M:%S")
    # mm/dd/YY H:M:S format
    return dt


def save_json(img_name,json_data,path_output):
    ## sp_name,image_ext = os.path.splitext(img_name)

    with open(os.path.join(path_output, img_name+'.json'), 'w') as outfile:
        json.dump(json_data, outfile)


def cv2_imshow_rgb(img,resize=None, figsize=(15,15)):
    if resize!=None:
        img=cv2.resize(img,resize)


    plt.figure(figsize=figsize)
    plt.imshow(np.uint8(img))


def display_multi(*images,resize=None, figsize=(15,15),bgr=False,axis=1):
    if resize!=None:
        res = np.array(cv2.resize(images[0],resize))
    else:
        res = np.array(images[0])

    for i in range(1,len(images)):

        if resize!=None:
            res_img = np.array(cv2.resize(images[i],resize))
        else:
            res_img = np.array(images[i])

        res = np.concatenate((res, res_img), axis=axis)

    if bgr==True:
        res = cv2.cvtColor(res,cv2.COLOR_BGR2RGB)

    return cv2_imshow_rgb(res,resize=None, figsize=figsize)


def make_folder(in_path):
    if not os.path.isdir(in_path):
        os.mkdir(in_path)


def make_folders_multi(*in_list):
    for in_path in in_list:
        make_folder(in_path)


def make_result_folders(in_path,rmv_folder=False):
    ''' generates empty directories to put drag results'''
    in_path_images = os.path.join(in_path,'intermediate_images/')

    if not os.path.isdir(in_path):
        os.mkdir(in_path)
        os.mkdir(in_path_images)
    elif os.path.isdir(in_path) and rmv_folder==True:
        shutil.rmtree(in_path)
        os.mkdir(in_path)
        os.mkdir(in_path_images)

    return in_path_images


def generate_video(video_path,inter_images_path,tar_points,res,res_points,out_ext,fps=24):
    '''generates a video of iterative movements of landmarks from source to destination '''
    for idx, (img, point) in enumerate(zip(res, res_points)):
        for p, t in zip(point, tar_points):
            red_patch = np.zeros([6, 6, 3])
            red_patch[..., 0] = np.ones([6, 6]) * 255
            blue_patch = np.zeros([6, 6, 3])
            blue_patch[..., 2] = np.ones([6, 6]) * 255

            img[p[1]-3:p[1]+3, p[0]-3:p[0]+3] = red_patch
            img[t[1]-3:t[1]+3, t[0]-3:t[0]+3] = blue_patch
        Image.fromarray(np.uint8(img)).save(f"{inter_images_path}/{idx+1}{out_ext}")

    os.system(f'ffmpeg -y -r {fps} -i "{inter_images_path}/%1d{out_ext}" -pix_fmt yuv420p -c:v libx264 "{video_path}"')


def display_video(filename,vid_res=(512,512)):
    '''displays a video in colab script '''

    video_data = io.open(filename, 'rb').read()
    video_b64 = base64.b64encode(video_data).decode('utf-8')
    html = f"""
    <video controls>
    <source src="data:video/mp4;base64,{video_b64}" type="video/mp4">
    </video>
    """
    html = f"""
    <video controls height={vid_res[0]} width={vid_res[1]}>
    <source src="data:video/mp4;base64,{video_b64}" type="video/mp4">
    </video>
    """
    display(HTML(html))


def generate_video_from_latents(out_vid_name,in_model,img_org,latent_in,latent_out,img_skip,
                                correct_eyes,eyes_mask,device,blur_kernel=(20,20),
                                num_images=30,video_fps=10,vid_resolution=(1024,1024)):

    inter_latents = np.linspace(latent_in.detach().cpu().numpy(),latent_out.detach().cpu().numpy(),num_images)

    write_encoding = 'XVID'
    fourcc = cv2.VideoWriter_fourcc(*write_encoding)
    video_writer = cv2.VideoWriter(out_vid_name, fourcc, video_fps, vid_resolution)

    for i in range(0,len(inter_latents)):
        inter_lat = torch.tensor(inter_latents[i]).to(device).unsqueeze(0)
        inter_img,_ = in_model.generator.synthesis.forward_v3(inter_lat,img_skip, update_emas=False,  noise_mode="const")
        inter_img = postprocess_image(inter_img.detach().cpu().numpy())
        if correct_eyes==True:
            inter_img = replace_parts(img_org,inter_img,eyes_mask,blur_kernel=blur_kernel)


        inter_img = cv2.resize(np.array(inter_img,dtype=np.uint8),vid_resolution)
        video_writer.write(inter_img[:,:,::-1])


def crop_and_make_square(image, detection_result, bbox_expand_ratio=0.2,padding_color=(255,255,255)):
    h, w, _ = image.shape

    # Extract face bounding box (assuming single face detection)
    if not detection_result.detections:
        return image  # Return original image if no face found
    
    detection = detection_result.detections[0]
    bbox = detection.bounding_box
    x, y, box_w, box_h = bbox.origin_x, bbox.origin_y, bbox.width, bbox.height

    # Expand bounding box
    expand_w = int(box_w * bbox_expand_ratio)
    expand_h = int(box_h * bbox_expand_ratio)

    x = max(0, x - expand_w // 2)
    y = max(0, y - expand_h // 2)
    box_w = min(w - x, box_w + expand_w)
    box_h = min(h - y, box_h + expand_h)

    # Compute new square size (max of width & height)
    max_dim = max(box_w, box_h)

    # Adjust cropping region to be square, prioritizing original image content
    extra_w = max_dim - box_w
    extra_h = max_dim - box_h

    # Try expanding within image bounds
    crop_x1 = max(0, x - extra_w // 2)
    crop_y1 = max(0, y - extra_h // 2)
    crop_x2 = min(w, crop_x1 + max_dim)
    crop_y2 = min(h, crop_y1 + max_dim)

    # Adjust crop if it exceeded image bounds
    crop_x1 = max(0, crop_x2 - max_dim)
    crop_y1 = max(0, crop_y2 - max_dim)

    # Crop the image
    cropped_image = image[crop_y1:crop_y2, crop_x1:crop_x2]

    # Check if padding is still needed
    final_h, final_w, _ = cropped_image.shape
    pad_x = max(0, max_dim - final_w)
    pad_y = max(0, max_dim - final_h)

    # Apply padding only if necessary
    if pad_x > 0 or pad_y > 0:
        cropped_image = cv2.copyMakeBorder(
            cropped_image,
            pad_y // 2, pad_y - pad_y // 2,
            pad_x // 2, pad_x - pad_x // 2,
            borderType=cv2.BORDER_CONSTANT,
            value=padding_color 
        )

    return cropped_image