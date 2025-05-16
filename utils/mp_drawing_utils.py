# Copyright 2020 The MediaPipe Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""MediaPipe solution drawing utils."""
import math
from typing import List, Mapping, Optional, Tuple, Union, Dict

import cv2
import dataclasses
import matplotlib.pyplot as plt
import numpy as np

from mediapipe.framework.formats import detection_pb2
from mediapipe.framework.formats import location_data_pb2
from mediapipe.framework.formats import landmark_pb2
import sys

from scipy.special import comb
from scipy.interpolate import interp1d
from scipy.interpolate import splprep, splev

# from utils.makeup_regions import forehead_landmarks
forehead_landmarks = {478:67, 479:109, 480:10, 481:338, 482:297, 483:103, 484:332}


_PRESENCE_THRESHOLD = 0.5
_VISIBILITY_THRESHOLD = 0.5
_RGB_CHANNELS = 3

WHITE_COLOR = (224, 224, 224)
BLACK_COLOR = (0, 0, 0)
RED_COLOR = (0, 0, 255)
GREEN_COLOR = (0, 128, 0)
BLUE_COLOR = (255, 0, 0)


@dataclasses.dataclass
class DrawingSpec:
  # Color for drawing the annotation. Default to the white color.
  color: Tuple[int, int, int] = WHITE_COLOR
  # Thickness for drawing the annotation. Default to 2 pixels.
  thickness: int = 2
  # Circle radius. Default to 2 pixels.
  circle_radius: int = 2
  # size
  size: int = 1


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


def draw_detection(
    image: np.ndarray,
    detection: detection_pb2.Detection,
    keypoint_drawing_spec: DrawingSpec = DrawingSpec(color=RED_COLOR),
    bbox_drawing_spec: DrawingSpec = DrawingSpec()):
  """Draws the detction bounding box and keypoints on the image.

  Args:
    image: A three channel RGB image represented as numpy ndarray.
    detection: A detection proto message to be annotated on the image.
    keypoint_drawing_spec: A DrawingSpec object that specifies the keypoints'
      drawing settings such as color, line thickness, and circle radius.
    bbox_drawing_spec: A DrawingSpec object that specifies the bounding box's
      drawing settings such as color and line thickness.

  Raises:
    ValueError: If one of the followings:
      a) If the input image is not three channel RGB.
      b) If the location data is not relative data.
  """
  if not detection.location_data:
    return
  if image.shape[2] != _RGB_CHANNELS:
    raise ValueError('Input image must contain three channel rgb data.')
  image_rows, image_cols, _ = image.shape

  location = detection.location_data
  if location.format != location_data_pb2.LocationData.RELATIVE_BOUNDING_BOX:
    raise ValueError(
        'LocationData must be relative for this drawing funtion to work.')
  # Draws keypoints.
  for keypoint in location.relative_keypoints:
    keypoint_px = _normalized_to_pixel_coordinates(keypoint.x, keypoint.y,
                                                   image_cols, image_rows)
    cv2.circle(image, keypoint_px, keypoint_drawing_spec.circle_radius,
               keypoint_drawing_spec.color, keypoint_drawing_spec.thickness)
  # Draws bounding box if exists.
  if not location.HasField('relative_bounding_box'):
    return
  relative_bounding_box = location.relative_bounding_box
  rect_start_point = _normalized_to_pixel_coordinates(
      relative_bounding_box.xmin, relative_bounding_box.ymin, image_cols,
      image_rows)
  rect_end_point = _normalized_to_pixel_coordinates(
      relative_bounding_box.xmin + relative_bounding_box.width,
      relative_bounding_box.ymin + relative_bounding_box.height, image_cols,
      image_rows)
  cv2.rectangle(image, rect_start_point, rect_end_point,
                bbox_drawing_spec.color, bbox_drawing_spec.thickness)


def draw_landmarks(
    image: np.ndarray,
    landmark_list: landmark_pb2.NormalizedLandmarkList,
    connections: Optional[List[Tuple[int, int]]] = None,
    landmark_drawing_spec: Union[DrawingSpec,
                                 Mapping[int, DrawingSpec]] = DrawingSpec(
                                     color=RED_COLOR),
    connection_drawing_spec: Union[DrawingSpec,
                                   Mapping[Tuple[int, int],
                                           DrawingSpec]] = DrawingSpec()):
  """Draws the landmarks and the connections on the image.

  Args:
    image: A three channel RGB image represented as numpy ndarray.
    landmark_list: A normalized landmark list proto message to be annotated on
      the image.
    connections: A list of landmark index tuples that specifies how landmarks to
      be connected in the drawing.
    landmark_drawing_spec: Either a DrawingSpec object or a mapping from
      hand landmarks to the DrawingSpecs that specifies the landmarks' drawing
      settings such as color, line thickness, and circle radius.
      If this argument is explicitly set to None, no landmarks will be drawn.
    connection_drawing_spec: Either a DrawingSpec object or a mapping from
      hand connections to the DrawingSpecs that specifies the
      connections' drawing settings such as color and line thickness.
      If this argument is explicitly set to None, no landmark connections will
      be drawn.

  Raises:
    ValueError: If one of the followings:
      a) If the input image is not three channel RGB.
      b) If any connetions contain invalid landmark index.
  """
  if not landmark_list:
    return
  if image.shape[2] != _RGB_CHANNELS:
    raise ValueError('Input image must contain three channel rgb data.')
  image_rows, image_cols, _ = image.shape
  idx_to_coordinates = {}
  for idx, landmark in enumerate(landmark_list.landmark):
    if ((landmark.HasField('visibility') and
         landmark.visibility < _VISIBILITY_THRESHOLD) or
        (landmark.HasField('presence') and
         landmark.presence < _PRESENCE_THRESHOLD)):
      continue
    landmark_px = _normalized_to_pixel_coordinates(landmark.x, landmark.y,
                                                   image_cols, image_rows)
    if landmark_px:
      idx_to_coordinates[idx] = landmark_px
  if connections:
    num_landmarks = len(landmark_list.landmark)
    # Draws the connections if the start and end landmarks are both visible.
    for connection in connections:
      start_idx = connection[0]
      end_idx = connection[1]
      if not (0 <= start_idx < num_landmarks and 0 <= end_idx < num_landmarks):
        raise ValueError(f'Landmark index is out of range. Invalid connection '
                         f'from landmark #{start_idx} to landmark #{end_idx}.')
      if start_idx in idx_to_coordinates and end_idx in idx_to_coordinates:
        drawing_spec = connection_drawing_spec[connection] if isinstance(
            connection_drawing_spec, Mapping) else connection_drawing_spec
        cv2.line(image, idx_to_coordinates[start_idx],
                 idx_to_coordinates[end_idx], drawing_spec.color,
                 drawing_spec.thickness)
  # Draws landmark points after finishing the connection lines, which is
  # aesthetically better.
  if landmark_drawing_spec:
    for idx, landmark_px in idx_to_coordinates.items():
      drawing_spec = landmark_drawing_spec[idx] if isinstance(
          landmark_drawing_spec, Mapping) else landmark_drawing_spec
      # White circle border
      circle_border_radius = max(drawing_spec.circle_radius + 1,
                                 int(drawing_spec.circle_radius * 1.2))
      cv2.circle(image, landmark_px, circle_border_radius, WHITE_COLOR,
                 drawing_spec.thickness)
      # Fill color into the circle
      cv2.circle(image, landmark_px, drawing_spec.circle_radius,
                 drawing_spec.color, drawing_spec.thickness)

      # Write landmark id
      # cv2.putText(image, org=landmark_px,text=str(idx),fontScale=0.8,fontFace = cv2.FONT_HERSHEY_SCRIPT_SIMPLEX,
      #            color=(255,0,0), thickness=2)
      cv2.putText(image, org=landmark_px,text=str(idx),fontScale=landmark_drawing_spec.size,fontFace = cv2.FONT_HERSHEY_SCRIPT_SIMPLEX,
                 color=landmark_drawing_spec.color, thickness=landmark_drawing_spec.thickness)



def draw_axis(
    image: np.ndarray,
    rotation: np.ndarray,
    translation: np.ndarray,
    focal_length: Tuple[float, float] = (1.0, 1.0),
    principal_point: Tuple[float, float] = (0.0, 0.0),
    axis_length: float = 0.1,
    axis_drawing_spec: DrawingSpec = DrawingSpec()):
  """Draws the 3D axis on the image.

  Args:
    image: A three channel RGB image represented as numpy ndarray.
    rotation: Rotation matrix from object to camera coordinate frame.
    translation: Translation vector from object to camera coordinate frame.
    focal_length: camera focal length along x and y directions.
    principal_point: camera principal point in x and y.
    axis_length: length of the axis in the drawing.
    axis_drawing_spec: A DrawingSpec object that specifies the xyz axis
      drawing settings such as line thickness.

  Raises:
    ValueError: If one of the followings:
      a) If the input image is not three channel RGB.
  """
  if image.shape[2] != _RGB_CHANNELS:
    raise ValueError('Input image must contain three channel rgb data.')
  image_rows, image_cols, _ = image.shape
  # Create axis points in camera coordinate frame.
  axis_world = np.float32([[0, 0, 0], [1, 0, 0], [0, 1, 0], [0, 0, 1]])
  axis_cam = np.matmul(rotation, axis_length*axis_world.T).T + translation
  x = axis_cam[..., 0]
  y = axis_cam[..., 1]
  z = axis_cam[..., 2]
  # Project 3D points to NDC space.
  fx, fy = focal_length
  px, py = principal_point
  x_ndc = np.clip(-fx * x / (z + 1e-5) + px, -1., 1.)
  y_ndc = np.clip(-fy * y / (z + 1e-5) + py, -1., 1.)
  # Convert from NDC space to image space.
  x_im = np.int32((1 + x_ndc) * 0.5 * image_cols)
  y_im = np.int32((1 - y_ndc) * 0.5 * image_rows)
  # Draw xyz axis on the image.
  origin = (x_im[0], y_im[0])
  x_axis = (x_im[1], y_im[1])
  y_axis = (x_im[2], y_im[2])
  z_axis = (x_im[3], y_im[3])
  cv2.arrowedLine(image, origin, x_axis, RED_COLOR, axis_drawing_spec.thickness)
  cv2.arrowedLine(image, origin, y_axis, GREEN_COLOR,
                  axis_drawing_spec.thickness)
  cv2.arrowedLine(image, origin, z_axis, BLUE_COLOR,
                  axis_drawing_spec.thickness)


def _normalize_color(color):
  return tuple(v / 255. for v in color)


def plot_landmarks(landmark_list: landmark_pb2.NormalizedLandmarkList,
                   connections: Optional[List[Tuple[int, int]]] = None,
                   landmark_drawing_spec: DrawingSpec = DrawingSpec(
                       color=RED_COLOR, thickness=5),
                   connection_drawing_spec: DrawingSpec = DrawingSpec(
                       color=BLACK_COLOR, thickness=5),
                   elevation: int = 10,
                   azimuth: int = 10):
  """Plot the landmarks and the connections in matplotlib 3d.

  Args:
    landmark_list: A normalized landmark list proto message to be plotted.
    connections: A list of landmark index tuples that specifies how landmarks to
      be connected.
    landmark_drawing_spec: A DrawingSpec object that specifies the landmarks'
      drawing settings such as color and line thickness.
    connection_drawing_spec: A DrawingSpec object that specifies the
      connections' drawing settings such as color and line thickness.
    elevation: The elevation from which to view the plot.
    azimuth: the azimuth angle to rotate the plot.
  Raises:
    ValueError: If any connetions contain invalid landmark index.
  """
  if not landmark_list:
    return
  plt.figure(figsize=(10, 10))
  ax = plt.axes(projection='3d')
  ax.view_init(elev=elevation, azim=azimuth)
  plotted_landmarks = {}
  for idx, landmark in enumerate(landmark_list.landmark):
    if ((landmark.HasField('visibility') and
         landmark.visibility < _VISIBILITY_THRESHOLD) or
        (landmark.HasField('presence') and
         landmark.presence < _PRESENCE_THRESHOLD)):
      continue
    ax.scatter3D(
        xs=[-landmark.z],
        ys=[landmark.x],
        zs=[-landmark.y],
        color=_normalize_color(landmark_drawing_spec.color[::-1]),
        linewidth=landmark_drawing_spec.thickness)
    plotted_landmarks[idx] = (-landmark.z, landmark.x, -landmark.y)
  if connections:
    num_landmarks = len(landmark_list.landmark)
    # Draws the connections if the start and end landmarks are both visible.
    for connection in connections:
      start_idx = connection[0]
      end_idx = connection[1]
      if not (0 <= start_idx < num_landmarks and 0 <= end_idx < num_landmarks):
        raise ValueError(f'Landmark index is out of range. Invalid connection '
                         f'from landmark #{start_idx} to landmark #{end_idx}.')
      if start_idx in plotted_landmarks and end_idx in plotted_landmarks:
        landmark_pair = [
            plotted_landmarks[start_idx], plotted_landmarks[end_idx]
        ]
        ax.plot3D(
            xs=[landmark_pair[0][0], landmark_pair[1][0]],
            ys=[landmark_pair[0][1], landmark_pair[1][1]],
            zs=[landmark_pair[0][2], landmark_pair[1][2]],
            color=_normalize_color(connection_drawing_spec.color[::-1]),
            linewidth=connection_drawing_spec.thickness)
  plt.show()



def draw_landmarks_delta(
    image: np.ndarray,
    landmark_list: landmark_pb2.NormalizedLandmarkList,
    connections: Optional[List[Tuple[int, int]]] = None,
    landmark_drawing_spec: Union[DrawingSpec,
                                 Mapping[int, DrawingSpec]] = DrawingSpec(
                                     color=RED_COLOR),
    connection_drawing_spec: Union[DrawingSpec,
                                   Mapping[Tuple[int, int],
                                           DrawingSpec]] = DrawingSpec(),
    landmark_deltas: Dict=None,
    close_flag=True,
    bezier_flag='bezier',
    s=3.0,
    k=2,
    img_parsed=None
                                           ):

  try:
    if img_parsed.all()!=None:
      #forehead_landmarks = {478:67, 479:109, 480:10, 481:338, 482:297}
      forehead_flag=True
  except:
      forehead_flag=False

  # print('forehead_flag',forehead_flag)

  try:
      assert bezier_flag==False or bezier_flag=='bezier' or bezier_flag=='smooth_poly' or bezier_flag=='smooth_poly_slpev'
  except:
      print('Set valid value for bezier flag: True,{},{}'.format('bezier','smooth_poly','smooth_poly_slpev'))
      sys.exit()
      #raise ValueError()

  if landmark_deltas != None:
      set_keys = set(landmark_deltas.keys())
      set_connections = set(np.unique(list(connections)))
    #   if len(set_keys-set_connections)!=0 or len(set_connections-set_keys)!=0:
    #       print('missing keys:',set_keys-set_connections,set_connections-set_keys)
    #       raise ValueError('The ids for all the landmarks must be same in connections and landmark_deltas')

      if len(set_keys-set_connections)!=0:
          print('missing keys:',set_keys-set_connections,set_connections-set_keys)
          raise ValueError('The ids for all the landmarks must be same in connections and landmark_deltas')

  if not landmark_list:
    return
  if image.shape[2] != _RGB_CHANNELS:
    raise ValueError('Input image must contain three channel rgb data.')
  image_rows, image_cols, _ = image.shape
  idx_to_coordinates = {}
  for idx, landmark in enumerate(landmark_list.landmark):
    if ((landmark.HasField('visibility') and
         landmark.visibility < _VISIBILITY_THRESHOLD) or
        (landmark.HasField('presence') and
         landmark.presence < _PRESENCE_THRESHOLD)):
      continue
    landmark_px = _normalized_to_pixel_coordinates(landmark.x, landmark.y,
                                                   image_cols, image_rows)

    if landmark_px:
      if landmark_deltas != None:
        if idx in landmark_deltas.keys():
          start_px_x = round(landmark_px[0]+image_cols*landmark_deltas[idx][0])
          start_px_y = round(landmark_px[1]+image_rows*landmark_deltas[idx][1])
          landmark_px = (start_px_x,start_px_y)

      idx_to_coordinates[idx] = landmark_px

  if forehead_flag==True:
    for idx in forehead_landmarks.values():
      inv_idx = (list(forehead_landmarks.keys())[list(forehead_landmarks.values()).index(idx)])


      landmark = landmark_list.landmark[idx]
      landmark_px = _normalized_to_pixel_coordinates(landmark.x, landmark.y,
                                                   image_cols, image_rows)

      y_idx_px = min(np.where(np.all(img_parsed[:,landmark_px[0],] == [255,85,0], axis=-1))[0])
      landmark_px = (landmark_px[0],y_idx_px)
      if landmark_px:
        if landmark_deltas != None:
          if inv_idx in landmark_deltas.keys():
            start_px_x = round(landmark_px[0]+image_cols*landmark_deltas[inv_idx][0])
            start_px_y = round(landmark_px[1]+image_rows*landmark_deltas[inv_idx][1])
            landmark_px = (start_px_x,start_px_y)

      landmark_px_f = landmark_px#(landmark_px[0],y_idx_px)
      idx_to_coordinates[inv_idx] = landmark_px_f


  bez_points = []
  if connections:
    if forehead_flag==True:
      num_landmarks = len(landmark_list.landmark)+len(forehead_landmarks.keys())
    else:
      num_landmarks = len(landmark_list.landmark)

    # Draws the connections if the start and end landmarks are both visible.
    for connection in connections:
      start_idx = connection[0]
      end_idx = connection[1]
      if not (0 <= start_idx < num_landmarks and 0 <= end_idx < num_landmarks):
        raise ValueError(f'Landmark index is out of range. Invalid connection '
                         f'from landmark #{start_idx} to landmark #{end_idx}.')
      if start_idx in idx_to_coordinates and end_idx in idx_to_coordinates:
        drawing_spec = connection_drawing_spec[connection] if isinstance(
            connection_drawing_spec, Mapping) else connection_drawing_spec

        if bezier_flag==False:
          cv2.line(image, idx_to_coordinates[start_idx],
                   idx_to_coordinates[end_idx], drawing_spec.color,
                   drawing_spec.thickness)
        else:
          bez_points.append(idx_to_coordinates[start_idx]) # only adding start idx coz the end idx will be the start idx in next iteration

    if close_flag==True and bezier_flag!=None:
      bez_points.append(idx_to_coordinates[end_idx])

    #print(bez_points)
    if bezier_flag=='bezier':
      # print(bezier_flag)
      xvals, yvals = bezier_curve(bez_points, nTimes=1000)
      for bz_i in range(0,len(xvals)-1):
        cv2.line(image, (round(xvals[bz_i]),round(yvals[bz_i])),
            (round(xvals[bz_i+1]),round(yvals[bz_i+1])), drawing_spec.color,
            drawing_spec.thickness)
    elif bezier_flag=='smooth_poly':
      # print(bezier_flag)
      xvals, yvals = smooth_poly(bez_points)
      for bz_i in range(0,len(xvals)-1):
        cv2.line(image, (round(xvals[bz_i]),round(yvals[bz_i])),
            (round(xvals[bz_i+1]),round(yvals[bz_i+1])), drawing_spec.color,
            drawing_spec.thickness)
    elif bezier_flag=='smooth_poly_slpev':
      # print(bezier_flag)
      xvals, yvals = smooth_poly_splev(bez_points,s=s,k=k)
      for bz_i in range(0,len(xvals)-1):
        cv2.line(image, (round(xvals[bz_i]),round(yvals[bz_i])),
            (round(xvals[bz_i+1]),round(yvals[bz_i+1])), drawing_spec.color,
            drawing_spec.thickness)


  # Draws landmark points after finishing the connection lines, which is
  # aesthetically better.
  if landmark_drawing_spec:
    for idx, landmark_px in idx_to_coordinates.items():
      drawing_spec = landmark_drawing_spec[idx] if isinstance(
          landmark_drawing_spec, Mapping) else landmark_drawing_spec
      # White circle border
      circle_border_radius = max(drawing_spec.circle_radius + 1,
                                 int(drawing_spec.circle_radius * 1.2))
      cv2.circle(image, landmark_px, circle_border_radius, WHITE_COLOR,
                 drawing_spec.thickness)
      # Fill color into the circle
      cv2.circle(image, landmark_px, drawing_spec.circle_radius,
                 drawing_spec.color, drawing_spec.thickness)

      # Write landmark id
      cv2.putText(image, org=landmark_px,text=str(idx),fontScale=0.3,fontFace = cv2.FONT_HERSHEY_SCRIPT_SIMPLEX,
                 color=(255,0,0), thickness=1)


def bernstein_poly(i, n, t):
    return comb(n, i) * ( t**(n-i) ) * (1 - t)**i


def bezier_curve(points, nTimes=1000):

    nPoints = len(points)
    xPoints = np.array([p[0] for p in points])
    yPoints = np.array([p[1] for p in points])

    t = np.linspace(0.0, 1.0, nTimes)

    polynomial_array = np.array([bernstein_poly(i, nPoints-1, t) for i in range(0, nPoints)])

    xvals = np.dot(xPoints, polynomial_array)
    yvals = np.dot(yPoints, polynomial_array)

    return xvals, yvals



def smooth_poly(points):
    x = [p[0] for p in points]
    y = [p[1] for p in points]

    t = np.arange(len(x))
    ti = np.linspace(0, t.max(), 5000)

    xi = interp1d(t, x, kind='cubic')(ti)
    yi = interp1d(t, y, kind='cubic')(ti)

    return xi,yi

def smooth_poly_splev(points,s=3.0,k=2,num_pts=1000):
    x = [p[0] for p in points]
    y = [p[1] for p in points]
    ##s = 3.0#3.0 # smoothness parameter
    ##k = 2 # spline order
    nest = -1 # estimate of number of knots needed (-1 = maximal)
    t, u = splprep([x, y], s=s, k=k, nest=-1)
    xn, yn = splev(np.linspace(0, 1, num_pts), t)

    return xn,yn
