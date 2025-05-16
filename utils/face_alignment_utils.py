## code for image FFHQ alignment
import scipy
import scipy.ndimage
import PIL
from PIL import Image
import numpy as np

def get_landmarks_dlib(img, predictor, detector):

    dets = detector(img, 1)

    for k, d in enumerate(dets):
        shape = predictor(img, d)

    t = list(shape.parts())
    a = []
    for tt in t:
        a.append([tt.x, tt.y])
    lm = np.array(a)
    return lm

def landmark_to_params(eye_left,eye_right,mouth_left,mouth_right):

    eye_avg = np.mean((eye_left,eye_right),axis=0) ##(eye_left + eye_right) * 0.5
    eye_to_eye = np.subtract(eye_left,eye_right)#eye_left-eye_right #eye_right - eye_left
    
    mouth_avg = np.mean((mouth_left,mouth_right),axis=0)#(mouth_left + mouth_right) * 0.5
    eye_to_mouth = mouth_avg - eye_avg

    # Choose oriented crop rectangle.
    x = eye_to_eye - np.flipud(eye_to_mouth) * [-1, 1]
    x /= np.hypot(*x)
    x *= max(np.hypot(*eye_to_eye) * 2.0, np.hypot(*eye_to_mouth) * 1.8)
    y = np.flipud(x) * [-1, 1]
    c = eye_avg + eye_to_mouth * 0.1
    quad = np.stack([c - x - y, c - x + y, c + x + y, c + x - y])
    qsize = np.hypot(*x) * 2

    return quad,qsize

def align_face(img, eye_left,eye_right,mouth_left,mouth_right, pad_value,transform_size=4096,output_size=1024,enable_padding=True):

    quad,qsize = landmark_to_params(eye_left,eye_right,mouth_left,mouth_right)

    # Shrink.
    shrink = int(np.floor(qsize / output_size * 0.5))
    if shrink > 1:
        rsize = (int(np.rint(float(np.shape(img)[0]) / shrink)), int(np.rint(float(np.shape(img)[1]) / shrink)))
        img = img.resize(rsize, PIL.Image.ANTIALIAS)
        quad /= shrink
        qsize /= shrink

    # Crop.
    border = max(int(np.rint(qsize * 0.1)), 3)
    # crop = (int(np.floor(min(quad[:, 0]))), int(np.floor(min(quad[:, 1]))), int(np.ceil(max(quad[:, 0]))),
    #         int(np.ceil(max(quad[:, 1]))))
    # crop = (max(crop[0] - border, 0), max(crop[1] - border, 0), min(crop[2] + border, np.shape(img)[0]),
    #         min(crop[3] + border, np.shape(img)[1]))
    # if crop[2] - crop[0] < np.shape(img)[0] or crop[3] - crop[1] < np.shape(img)[1]:
    #     img = img.crop(crop)
    #     quad -= crop[0:2]

        
    # Pad.
    pad = (int(np.floor(min(quad[:, 0]))), int(np.floor(min(quad[:, 1]))), int(np.ceil(max(quad[:, 0]))),
           int(np.ceil(max(quad[:, 1]))))
    pad = (max(-pad[0] + border, 0), max(-pad[1] + border, 0), max(pad[2] - np.shape(img)[0] + border, 0),
           max(pad[3] - np.shape(img)[1] + border, 0))
    if enable_padding and max(pad) > border - 4:
        pad = np.maximum(pad, int(np.rint(qsize * 0.3)))

        if pad_value['mode'] == 'constant':
            cval = np.array([[pad_value['color'], pad_value['color']], [pad_value['color'], pad_value['color']], [0, 0]], dtype=object)
            img = np.pad(np.float32(img), ((pad[1], pad[3]), (pad[0], pad[2]), (0, 0)), 'constant',constant_values=cval)
        else:
            img = np.pad(np.float32(img), ((pad[1], pad[3]), (pad[0], pad[2]), (0, 0)), pad_value['mode'])

        h, w, _ = img.shape
        y, x, _ = np.ogrid[:h, :w, :1]
        mask = np.maximum(1.0 - np.minimum(np.float32(x) / pad[0], np.float32(w - 1 - x) / pad[2]),
                          1.0 - np.minimum(np.float32(y) / pad[1], np.float32(h - 1 - y) / pad[3]))
        blur = qsize * 0.02
        img += (scipy.ndimage.gaussian_filter(img, [blur, blur, 0]) - img) * np.clip(mask * 3.0 + 1.0, 0.0, 1.0)
        img += (np.median(img, axis=(0, 1)) - img) * np.clip(mask, 0.0, 1.0)
        img = PIL.Image.fromarray(np.uint8(np.clip(np.rint(img), 0, 255)), 'RGB')
        quad += pad[:2]

    # Transform.
    img = img.transform((transform_size, transform_size), PIL.Image.QUAD, (quad + 0.5).flatten(), PIL.Image.BILINEAR)
    if output_size < transform_size:
        img = img.resize((output_size, output_size))

    # Return aligned image.
    return img

