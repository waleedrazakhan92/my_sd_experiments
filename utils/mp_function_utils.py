import mediapipe as mp
import math
from typing import Tuple, Union
import utils.mp_drawing_utils as mp_drawing
import cv2
import numpy as np
from mediapipe.framework.formats.landmark_pb2 import NormalizedLandmark, NormalizedLandmarkList

mp_face_mesh = mp.solutions.face_mesh
annotated_image_rescale = 4
mp_drawing_styles = mp.solutions.drawing_styles
drawing_spec = mp_drawing.DrawingSpec(thickness=2, circle_radius=3, color=[255,255,0])
drawing_spec_landmarks = mp_drawing.DrawingSpec(thickness=1, circle_radius=1, color=[255,0,0], size=0.1*annotated_image_rescale) #size=0.2

mp_face_mesh = mp.solutions.face_mesh

def _normalized_to_pixel_coordinates(
    normalized_x: float, normalized_y: float, image_width: int,
    image_height: int) -> Union[None, Tuple[int, int]]:
  """Converts normalized value pair to pixel coordinates."""

  # Checks if the float value is between 0 and 1.
  def is_valid_normalized_value(value: float) -> bool:
    return (value > 0 or math.isclose(0, value)) and (value < 1 or
                                                      math.isclose(1, value))

  if not (is_valid_normalized_value(normalized_x) and
          is_valid_normalized_value(normalized_y)):
    # TODO: Draw coordinates even if it's outside of the image bounds.
    return None
  x_px = min(math.floor(normalized_x * image_width), image_width - 1)
  y_px = min(math.floor(normalized_y * image_height), image_height - 1)
  return x_px, y_px

def convert_landmark_to_px(landmark,img_landmarks,image_cols,image_rows,get_z=False):
    landmark = img_landmarks.multi_face_landmarks[0].landmark[landmark]
    landmark_px = _normalized_to_pixel_coordinates(landmark.x, landmark.y,
                                                   image_cols, image_rows)
    if get_z==True:
        landmark_px = landmark_px+(landmark.z,)

    return landmark_px

def mp_idx_2_fixed_px(img_landmarks,fixed_indices,image_cols,image_rows,get_z=False):
    '''
    converts the mediapipe landmark indices for fixed before/afters to list of pixels for drag process
    '''
    fixed_px_list = []
    for i in range(0,len(fixed_indices)):
        land_px = convert_landmark_to_px(fixed_indices[i],img_landmarks,image_cols,image_rows,get_z=get_z)
        # fixed_px_list.append(np.array(land_px))
        fixed_px_list.append(land_px)

    return fixed_px_list

def landmark_list_to_tuple(in_landmarks):
    '''
    convert landmarks list to list of tuples as draw_landmarks function takes list of tuples
    '''
    out_landmarks = []
    for i in range(0,len(in_landmarks)-1):
        out_landmarks.append((in_landmarks[i],in_landmarks[i+1]))

    out_landmarks.append((in_landmarks[-1],in_landmarks[0]))

    return frozenset(out_landmarks)

def find_landmarks(img, max_faces=1, min_confidence=0.5):
    # Run MediaPipe Face Mesh.
    with mp_face_mesh.FaceMesh(
        static_image_mode=True,
        refine_landmarks=True,
        max_num_faces=max_faces,
        min_detection_confidence=min_confidence) as face_mesh:

        results = face_mesh.process(img)

    return results

def display_mask_over_image(img,mask,mask_weight=0.3):
    mask = cv2.cvtColor(mask, cv2.COLOR_GRAY2BGR)
    return cv2.addWeighted(img, (1-mask_weight), mask, mask_weight, 0)

# def make_mask(img, img_landmarks, connections,thickness=3):
#     drawing_spec.thickness=thickness
#     mask = np.zeros(np.shape(img), np.uint8)
#     mp_drawing.draw_landmarks(
#                 image=mask,
#                 landmark_list=img_landmarks.multi_face_landmarks[0],
#                 connections=connections,
#                 landmark_drawing_spec=None,
#                 connection_drawing_spec=drawing_spec)

#     return mask

def make_mask(img,img_landmarks,connections):
    img_rows,img_cols = np.shape(img)[:2]
    points = mp_idx_2_fixed_px(img_landmarks,connections,img_cols,img_rows)
    mask = np.zeros((img_rows,img_cols), dtype=np.uint8)
    cv2.fillPoly(mask, [np.array(points)], 255)
    return mask

def fill_mask(mask):
    gray = cv2.cvtColor(mask, cv2.COLOR_RGB2GRAY)
    cnts = cv2.findContours(gray, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cnts = cnts[0] if len(cnts) == 2 else cnts[1]
    for c in cnts:
        cv2.drawContours(gray,[c], 0, (255,255,255), -1)

    return gray


def dilate_mask(mask,dilate_iters=1,kernel_shape=(10,10)):
    # kernel = np.ones((5,5), np.uint8)
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE,kernel_shape)
    dilated_mask = cv2.dilate(mask, kernel, iterations=dilate_iters)
    return dilated_mask

def combine_mask(img,img_landmarks,connections):
    combined_mask = np.zeros(np.shape(img)[0:2], np.uint8)
    for i in range(0,len(connections)):
        mask = make_mask(img, img_landmarks, connections)
        mask = fill_mask(mask)
        combined_mask = cv2.bitwise_or(combined_mask, mask)

    return combined_mask

def draw_landmarks_on_image(land_img,img_landmarks,in_landmarks,thickness=3):
    drawing_spec.thickness=thickness
    mp_drawing.draw_landmarks_delta(
                image=land_img,
                landmark_list=img_landmarks.multi_face_landmarks[0],
                connections=landmark_list_to_tuple(in_landmarks),
                landmark_drawing_spec=None,
                connection_drawing_spec=drawing_spec,
                landmark_deltas=None,
                        close_flag=False,bezier_flag=False,
                        s=3,k=2,img_parsed=False)

    return land_img

def extend_mp_landmarks(mp_landmarks,extra_landmarks):
    for face_landmarks in mp_landmarks.multi_face_landmarks:
        # Convert immutable NormalizedLandmarkList to a mutable list
        landmarks_list = [{"x": lm.x, "y": lm.y, "z": lm.z} for lm in face_landmarks.landmark]

        # Append extra landmarks
        landmarks_list.extend(extra_landmarks)

        updated_landmark_list = NormalizedLandmarkList(
            landmark=[NormalizedLandmark(x=lm["x"], y=lm["y"], z=lm["z"]) for lm in landmarks_list]
        )

    return updated_landmark_list
