## mediapipe and masking
from utils.mp_function_utils import make_mask,dilate_mask
import cv2
import numpy as np
from utils.misc_VGGH_utils import generate_mask

def replace_parts(real_img,in_image,in_mask,blur_kernel=(50,50)):
    '''
    replace parts of one image with another using mask generated from mediapipe landmarks
    '''
    final_img = np.array(in_image).copy()
    mask_3d = cv2.merge([in_mask, in_mask, in_mask])
    mask_3d = cv2.blur(mask_3d, blur_kernel)
    final_img = final_img*(1-mask_3d) + real_img*(mask_3d)
    return final_img

def get_desired_mask_mp(img,img_landmarks,part_landmarks,dilate_iters=5,blur_kernel=(20,20)):
    '''
    generate a mask using the mediapipe landmarks
    '''
    in_mask = make_mask(img, img_landmarks, connections=part_landmarks)
    in_mask = dilate_mask(in_mask,dilate_iters=dilate_iters,kernel_shape=(5,5))
    in_mask = in_mask/255
    in_mask = cv2.blur(in_mask, blur_kernel)
    return in_mask

def generate_mask_from_landmarks(in_img,img_landmarks,all_masked_parts,all_dilations):
    '''
    generate mask using a list of mediapipe landmarks
    '''
    assert (len(all_dilations)==1) or (len(all_masked_parts)==len(all_dilations)),'length of all maskes and dilations should be same or dilations len(1)'

    if len(all_dilations)==1:
        dilation_list = all_dilations*len(all_masked_parts)
    else:
        dilation_list = all_dilations

    all_masks = []
    for p in range(0,len(all_masked_parts)):
        in_parts = all_masked_parts[p]
        # in_parts = landmark_list_to_tuple(in_parts)

        current_mask = get_desired_mask_mp(in_img,img_landmarks,in_parts,dilate_iters=dilation_list[p],blur_kernel=(1,1))
        all_masks.append(current_mask)
        if p==0:
            img_mask = np.zeros_like(current_mask)

        img_mask = np.clip(cv2.bitwise_or(img_mask,current_mask),0,1)

    return img_mask,all_masks


## masking for a x,y,z type landmarks (not necesserely mediapipe)
def get_desired_mask_simple(img,img_landmarks,part_landmarks,dilate_iters=5,blur_kernel=(20,20)):
    '''
    generate a mask using the mediapipe landmarks
    '''
    # in_mask = make_mask(img, img_landmarks, connections=part_landmarks)
    in_mask = generate_mask(np.array(img),np.squeeze(img_landmarks)[:,0:2][part_landmarks])
    in_mask = dilate_mask(in_mask,dilate_iters=dilate_iters,kernel_shape=(5,5))
    in_mask = in_mask/255
    in_mask = cv2.blur(in_mask, blur_kernel)
    return in_mask

def generate_mask_from_landmarks_simple(in_img,img_landmarks,all_masked_parts,all_dilations):
    '''
    generate mask using a list of landmarks
    '''
    assert (len(all_dilations)==1) or (len(all_masked_parts)==len(all_dilations)),'length of all maskes and dilations should be same or dilations len(1)'

    if len(all_dilations)==1:
        dilation_list = all_dilations*len(all_masked_parts)
    else:
        dilation_list = all_dilations

    all_masks = []
    for p in range(0,len(all_masked_parts)):
        in_parts = all_masked_parts[p]
        # in_parts = landmark_list_to_tuple(in_parts)

        current_mask = get_desired_mask_simple(in_img,img_landmarks,in_parts,dilate_iters=dilation_list[p],blur_kernel=(1,1))
        all_masks.append(current_mask)
        if p==0:
            img_mask = np.zeros_like(current_mask)

        img_mask = np.clip(cv2.bitwise_or(img_mask,current_mask),0,1)

    return img_mask,all_masks

