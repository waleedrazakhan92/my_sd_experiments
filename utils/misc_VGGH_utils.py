import matplotlib.pyplot as plt
import numpy as np
import cv2
from utils.mp_function_utils import extend_mp_landmarks

def find_and_extend_landmarks(vgg_heads_model,img,mp_landmarks,in_landmarks):
    img_rows,img_cols = np.shape(img)[:2]
    vggh_preds = vgg_heads_model(np.array(img))
    extra_landmarks = get_vggh_subset(vggh_preds,in_landmarks,img_rows,img_cols)
    extended_mp = extend_mp_landmarks(mp_landmarks,extra_landmarks)
    mp_landmarks.multi_face_landmarks[0] = extended_mp
    return mp_landmarks,vggh_preds

def generate_mask(img_array, clicked_points):
    height, width = img_array.shape[:2]
    mask = np.zeros((height, width), dtype=np.uint8)

    # Ensure at least 3 points to form a polygon
    if len(clicked_points) < 3:
        print("Need at least 3 points to form an enclosed shape.")
        return mask

    # Convert clicked_points to a format suitable for OpenCV
    polygon = np.array([clicked_points], dtype=np.int32)

    # Fill the polygon on the mask
    cv2.fillPoly(mask, polygon, 255)  # Fill with white (255)

    return mask

def on_click(event, clicked_points, ax, fig):
    if event.xdata is not None and event.ydata is not None:  # Ensure valid click
        x, y = int(event.xdata), int(event.ydata)
        clicked_points.append((x, y))
        print(f"Clicked point: ({x}, {y})")

        # Update the plot with the clicked point
        ax.plot(x, y, 'ro',markersize=2)  # Red dot
        fig.canvas.draw()

def select_landmarks(img_array,clicked_points,figsize=(10,10)):
    
    fig, ax = plt.subplots(figsize=figsize)
    ax.imshow(img_array)
    ax.set_title("Click on the image to select points. Close the plot when done.")

    # Connect the click event
    fig.canvas.mpl_connect('button_press_event', lambda event: on_click(event, clicked_points, ax, fig))

    # Show the image and wait for user interaction
    plt.show()


def get_nearest_landmarks(image: np.ndarray, predictions: np.ndarray, coordinates: list, n: int, color: tuple = (255, 0, 0), radius: int = 3, write_idx=True):
    """
    Draw the nearest `n` prediction landmarks to a given list of coordinates on an image.

    Args:
        image (np.ndarray): The input image on which landmarks will be drawn.
        predictions (np.ndarray): A (N, 3) array containing predicted landmark coordinates (x, y, z).
        coordinates (list): A list of (x, y) tuples representing target coordinates.
        n (int): Number of nearest landmarks to find for each target coordinate.
        color (tuple): Color of the landmark points and text, default is blue (BGR format).
        radius (int): Radius of the landmark points, default is 3 pixels.

    Returns:
        np.ndarray: The image with landmarks drawn.
        List[List[int]]: A list of nearest landmarks indices for each coordinate.
    """
    # Make a copy of the image to avoid modifying the original
    output_image = image.copy()

    all_nearest_indices = []
    for idx, (coord_x, coord_y) in enumerate(coordinates):
        distances = []

        # Calculate the distance to each prediction
        for i, (pred_x, pred_y, pred_z) in enumerate(predictions):
            if pred_z<0:  ## z<0 are the landmarks from the front
                distance = np.sqrt((coord_x - pred_x) ** 2 + (coord_y - pred_y) ** 2)
                distances.append((distance, i))

        # Sort distances and get the nearest `n` indices
        distances.sort(key=lambda x: x[0])
        nearest_indices = [i for _, i in distances[:n]]

        # Add nearest indices to the result list
        all_nearest_indices.append(nearest_indices)

        # Draw the nearest landmarks and their indices
        for nearest_index in nearest_indices:
            nearest_x, nearest_y, _ = predictions[nearest_index]
            cv2.circle(output_image, (int(nearest_x), int(nearest_y)), radius, color, -1)
            if write_idx==True:
                cv2.putText(output_image, f"{nearest_index}", (int(nearest_x) + 5, int(nearest_y)), 
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1, cv2.LINE_AA)

        # Optionally draw the target coordinate (uncomment below if needed)
        cv2.circle(output_image, (int(coord_x), int(coord_y)), radius, (0, 255, 255), -1)

    return output_image, all_nearest_indices



# Function to filter landmarks by mask
def filter_landmarks_by_mask(mask, landmarks_3d):
    """
    Filters 3D landmarks to return only those lying within the masked area.

    Parameters:
    - mask (numpy.ndarray): A binary mask (2D array) with the same dimensions as the image.
    - landmarks_3d (numpy.ndarray): A 2D array of shape (N, 3), where each row is (x, y, z) for a predicted landmark.

    Returns:
    - filtered_landmarks (list): List of filtered landmarks .
    """
    filtered_landmarks = []

    for i,(x, y, z) in enumerate(landmarks_3d):
        # Resize landmark coordinates
        x_resized = int(x )
        y_resized = int(y )

        # Check if the landmark is within the mask bounds
        if 0 <= x_resized < mask.shape[1] and 0 <= y_resized < mask.shape[0]:
            if mask[y_resized, x_resized] > 0:  # Check if the landmark is within the masked area
                # filtered_landmarks.append((i,x_resized, y_resized, z))
                if z<0:
                    filtered_landmarks.append(i)


    return filtered_landmarks


def vggH_idx_2_pixels(img_landmarks,fixed_indices):
    '''
    converts the vgg heads landmark indices for fixed before/afters to list of pixels 
    '''
    fixed_px_list = []
    for i in fixed_indices:
        fixed_px_list.append((int(img_landmarks[i][0]),int(img_landmarks[i][1])))

    return fixed_px_list

def get_vggh_subset(vggh_landmarks,selected_indices,img_rows,img_cols):
    landmark_list = []
    for l in selected_indices:
        landmark_list.append({'x':vggh_landmarks.heads[0].vertices_3d[l][0]/img_cols,
                              'y':vggh_landmarks.heads[0].vertices_3d[l][1]/img_rows,
                              'z':vggh_landmarks.heads[0].vertices_3d[l][2]
                              })
        
    return landmark_list
    

# import io
# import base64
# from IPython.display import HTML, display

# %matplotlib widget

## -----------------------------------------------------
## for interactive landmark selection
## -----------------------------------------------------
# # for masked filtering of landmarks

# # Initialize empty list for clicked points
# clicked_points = []

# # Call the function to select landmarks
# select_landmarks(img)

# # Print the selected landmarks
# print("Selected landmarks:", clicked_points)


# img_mask = generate_mask(img, clicked_points)
# filtered_landmarks = filter_landmarks_by_mask(img_mask, predictions.heads[0].vertices_3d)
# img_filtered = draw_filtered_landmarks(filtered_landmarks,img.copy(),img_mask,resize_scale=2)

# # output_image = draw_landmarks(img,predictions.heads[0].vertices_3d,scale=1)
# # display_multi(output_image,figsize=(5,5))

# display_multi(img_filtered,figsize=(5,5))


# from utils.misc_utils import select_landmarks

# %matplotlib widget
# # Initialize empty list for clicked points
# clicked_points = []

# # Call the function to select landmarks
# select_landmarks(img,clicked_points)

# # Print the selected landmarks
# print("Selected landmarks:", clicked_points)


# from utils.misc_utils import get_nearest_landmarks
# nearest_indices = []
# output_image,nearest_indices = get_nearest_landmarks(img.copy(), predictions.heads[0].vertices_3d, clicked_points, n=2)
# print(nearest_indices)
# # print(predictions.heads[0].vertices_3d[nearest_indices])

# display_multi(output_image,figsize=(8,8))

## -----------------------------------------------------