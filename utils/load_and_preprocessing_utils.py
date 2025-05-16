import torch
from PIL import Image
import numpy as np
import shutil
import joblib
import numpy as np

def load_direction_joblib(dir_path):
    clf = joblib.load(dir_path)
    clf = clf.coef_.reshape((18, 512))
    assert clf.shape==(18,512),'direction shape should be (18,512)'
    return clf

def load_direction_npy(dir_path):
    dir_npy = np.load(dir_path)
    if dir_npy.shape==(1,512):
        dir_npy = np.repeat(dir_npy,18,axis=0)

    assert dir_npy.shape==(18,512),'direction shape should be (18,512)'
    return dir_npy

def load_direction(in_path):
    """Loads pretrained directions"""
    if in_path.endswith(".joblib"):
        return load_direction_joblib(in_path)
    elif in_path.endswith(".npy"):
        return load_direction_npy(in_path)
    else:
        raise ValueError(f"❌ Unsupported file format for {in_path}")


def load_img_pil(img_path, resize_dims=None):
    original_image = Image.open(img_path)
    original_image = original_image.convert("RGB")
    if resize_dims !=None:
        original_image = original_image.resize(resize_dims)

    return original_image

def preprocess_image(image):
    image = image.resize((256, 256))
    image = np.asarray(image).transpose(2, 0, 1).astype(np.float32) # C,H,W -> H,W,C
    image = torch.FloatTensor(image.copy())
    image = (image - 127.5) / 127.5     # Normalize
    return image.unsqueeze(0)

def postprocess_image(image, min_val=-1.0, max_val=1.0):
    """
        Input: A nwumpy image with shape NxCxHxW.
        Output: Output image with NxHxWxC with values between 0-255
    """
    image = image.astype(np.float64)
    image = (image - min_val) * 255 / (max_val - min_val)
    image = np.clip(image + 0.5, 0, 255).astype(np.uint8)
    image = image.transpose(0, 2, 3, 1)
    image = Image.fromarray(image[0])
    return image

def resize_big_img(img,resize_big):
    assert type(resize_big)==int, 'resize_big should be type int '

    im_h,im_w = img.size[0:2]
    if resize_big>1:
        ## just because its a big image
        img = img.resize((round(img.size[0]/resize_big),round(img.size[1]/resize_big)))
    elif resize_big==-1:
        resize_big_auto = 1024
        if min(im_h,im_w)>resize_big_auto:
            resize_big_auto = round(min(im_h,im_w)/resize_big_auto)
            img = img.resize((round(img.size[0]/resize_big_auto),round(img.size[1]/resize_big_auto)))

    return img


###########################
## loading multiple files
###########################
from multiprocessing import Pool, Process, Manager
# import numpy as np
import itertools

def load_lat_np(file):
    lat = np.load(file)
    return lat

def load_lat_multi(path_latents):
    pool = Pool(8)
    all_latents = pool.map(load_lat_np, path_latents)
    pool.close()
    pool.join()
    return all_latents


def move_file(in_file,cpy_path):
    shutil.copy(in_file, cpy_path)

def move_multi(path_latents, path_copy):
    pool = Pool(8)
    pool.starmap(move_file, zip(path_latents, itertools.repeat(path_copy)))
    pool.close()
    pool.join()

def copy_multi(path_latents, path_copy,pool_size=8):
    pool = Pool(pool_size)
    pool.starmap(move_file, zip(path_latents, itertools.repeat(path_copy)))
    pool.close()
    pool.join()
