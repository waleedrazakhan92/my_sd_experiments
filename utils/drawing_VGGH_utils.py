import cv2
import numpy as np

def draw_bounding_box(image, bbox):
    """
    Draws a bounding box on the given image.

    Parameters:
    image (numpy.ndarray): The image on which to draw the bounding box.
    bbox (object): The bounding box object with attributes x, y, w, h.

    Returns:
    numpy.ndarray: The image with the bounding box drawn.
    """
    start_point = (bbox.x, bbox.y)  # Top-left corner
    end_point = (bbox.x + bbox.w, bbox.y + bbox.h)  # Bottom-right corner
    color = (0, 255, 0)  # Green color in BGR
    thickness = 2  # Thickness of the box

    # Draw the rectangle on the image
    cv2.rectangle(image, start_point, end_point, color, thickness)

    return image

def draw_filtered_landmarks(filtered_landmarks,image,scale=1):

    img_h,img_w = image.shape[:2]
    image = cv2.resize(image,(img_w*scale,img_h*scale))
    # Draw the filtered landmarks on the resized image
    for i, (idx,x, y, z) in enumerate(filtered_landmarks):
        # Draw a small circle for the landmark
        cv2.circle(image, (x*scale, y*scale), radius=3, color=(0, 255, 0), thickness=-1)
        
        # Annotate the landmark index
        cv2.putText(image, str(idx), (x*scale + 3, y*scale), 
                    cv2.FONT_HERSHEY_SCRIPT_SIMPLEX, 0.5, (255, 0, 0), 1)

    return image



def draw_landmarks(image: np.ndarray, landmarks: np.ndarray, selected_indices: list = [],scale: float = 1.0, color_pt: tuple = (0, 255, 0),color_txt: tuple=(255,0,0),
                   radius: int = 3, fontScale: float = 0.5, thickness: int = 1,font=cv2.FONT_HERSHEY_SCRIPT_SIMPLEX, write_idx=True) -> np.ndarray:
    """
    Draw landmarks on an image with optional resizing.

    Args:
        image (np.ndarray): The input image on which landmarks will be drawn.
        landmarks (np.ndarray): A (N, 3) array containing landmark coordinates (x, y, z).
        scale (float): Scaling factor for resizing the image, default is 1.0 (no resizing).
        color (tuple): Color of the landmark points, default is green (BGR format).
        radius (int): Radius of the landmark points, default is 3 pixels.

    Returns:
        np.ndarray: The image with landmarks drawn.
    """
    
    image = image.copy()
    landmarks = landmarks.copy()
    if len(selected_indices)==0:
        selected_indices = list(range(len(landmarks)))#list(range(0,6000))

    # Resize the image if scale is not 1.0
    if scale != 1.0:
        height, width = image.shape[:2]
        image = cv2.resize(image, (int(width * scale), int(height * scale)))
        # landmarks[:, :2] *= scale  # Adjust the x and y coordinates of landmarks

    # Make a copy of the image to avoid modifying the original
    output_image = image.copy()

    for idx in selected_indices:
        landmark = landmarks[idx]
        x, y, z = landmark[0],landmark[1],landmark[2]
        if (z < 0) and (idx in selected_indices):
            cv2.circle(output_image, (int(x*scale), int(y*scale)), radius, color_pt, -1)
            if write_idx==True:
                cv2.putText(output_image, str(idx), (int(x*scale)+3, int(y*scale)), 
                            font, fontScale, color_txt, thickness, cv2.LINE_AA)


    return output_image


def draw_landmark_lines(image, landmark_pairs, predictions, color=(0, 255, 0), thickness=2):
    """
    Draws lines between landmark pairs on the given image.

    Parameters:
        image (numpy.ndarray): The image to draw on.
        landmark_pairs (list of tuples): A list of landmark pairs, e.g., [(13, 24), (22, 333)].
        predictions: Object containing landmark vertices (e.g., predictions.heads[0].vertices_3d).
        color (tuple): Line color in BGR format (default is green).
        thickness (int): Line thickness (default is 2).
    
    Returns:
        numpy.ndarray: The image with drawn lines.
    """
    image = image.copy()
    for point1_idx, point2_idx in landmark_pairs:
        # Get the 3D coordinates of the landmarks
        point1 = predictions.heads[0].vertices_3d[point1_idx]
        point2 = predictions.heads[0].vertices_3d[point2_idx]
        
        # Convert 3D points to 2D if needed (using x and y coordinates)
        x1, y1 = int(point1[0]), int(point1[1])
        x2, y2 = int(point2[0]), int(point2[1])
        
        # Draw the line on the image
        cv2.line(image, (x1, y1), (x2, y2), color, thickness)
    
    return image