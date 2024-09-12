import numpy as np
import torch
from scipy.ndimage import gaussian_filter, zoom
from skimage.feature import peak_local_max
from scipy.special import logsumexp
from PIL import Image
import matplotlib.pyplot as plt
import deepgaze_pytorch
import base64
from io import BytesIO
from matplotlib.colors import Normalize
from skimage.measure import label, regionprops


DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
model = deepgaze_pytorch.DeepGazeIIE(pretrained=True).to(DEVICE)
model.eval()

def preprocess_image(image_data):
    
    if image_data.startswith('data:image'):
        # Find the start of the base64 string
      base64_str_idx = image_data.find('base64,') + 7
      # Extract the base64 string
      image_data = image_data[base64_str_idx:]

    image_bytes = base64.b64decode(image_data)
    # Create a file-like object from bytes
    image_file = BytesIO(image_bytes)
    # Use Image.open to read the file-like object
    image = Image.open(image_file).convert('RGB')
    image = np.array(image)  # Load image and convert to numpy array
    centerbias_template = np.load('centerbias_mit1003.npy')  # Load centerbias
    # Rescale centerbias to match image dimensions
    centerbias = zoom(centerbias_template, (image.shape[0]/centerbias_template.shape[0], image.shape[1]/centerbias_template.shape[1]), order=0, mode='nearest')
    centerbias -= logsumexp(centerbias)  # Normalize centerbias
    # Convert to tensors
    image_tensor = torch.tensor([image.transpose(2, 0, 1)], dtype=torch.float32).to(DEVICE)
    centerbias_tensor = torch.tensor([centerbias], dtype=torch.float32).to(DEVICE)
    return image, image_tensor, centerbias_tensor

def predict_fixation(image_tensor, centerbias_tensor):
    with torch.no_grad():
        log_density_prediction = model(image_tensor, centerbias_tensor)
    return log_density_prediction

def calculate_highlighted_areas(density_map_normalized, intensity_threshold=0.5):
    # Threshold the density map to find highlighted areas
    highlighted_areas = density_map_normalized > intensity_threshold
    # Label distinct highlighted areas
    labeled_areas, num_features = label(highlighted_areas, return_num=True)
    # Calculate total highlighted area
    total_area = np.sum(highlighted_areas)

    return labeled_areas, num_features, total_area

#def plot_results(image, log_density_prediction):
#    density_map = log_density_prediction.detach().cpu().numpy()[0, 0]  # Convert tensor to numpy
#    fixation_points = find_fixation_points(density_map)  # Find fixation points
#    plt.imshow(image)  # Plot original image
#
#    # Assuming you have scalar values for each fixation point (e.g., intensities)
#    # For demonstration, let's use a simple range as scalar values
#    # intensities = range(len(fixation_points))
#    density_map_min = np.min(density_map)
#    density_map_max = np.max(density_map)
#    density_map_normalized = (density_map - density_map_min) / (density_map_max - density_map_min)
#    intensities = [density_map_normalized[int(y), int(x)] for y, x in fixation_points]
#
#    # Use the 'c' parameter to provide scalar values and 'cmap' for the colormap
#    plt.scatter(fixation_points[:, 1], fixation_points[:, 0], c=intensities, cmap='viridis')
#
#    # Draw lines connecting fixation points
#    for i, (y, x) in enumerate(fixation_points):
#        color = 'yellow' if i == 0 else 'red'
#        plt.scatter(x, y, c=intensities,  color=color)  # Mark fixation point with color
#        if i > 0:
#            prev_y, prev_x = fixation_points[i-1]
#            plt.plot([prev_x, x], [prev_y, y], color=color, linewidth=2)  # Draw line in red
#
#    plt.show()

def find_fixation_points(density_map, num_points=3):
    smoothed_map = gaussian_filter(density_map, sigma=3)  # Apply Gaussian blur
    coordinates = peak_local_max(smoothed_map, num_peaks=num_points, min_distance=20)  # Find local maxima
    intensities = smoothed_map[coordinates[:, 0], coordinates[:, 1]]
    sorted_indices = np.argsort(intensities)[::-1]  # Sort by intensity
    sorted_coords = coordinates[sorted_indices]
    return sorted_coords


def generate_base64_heatmap(image_path, color = 'plasma', min_points = 3, max_points= None, show_fixation_points = True):
    try: 
    # Preprocess the image and get tensors
        image, image_tensor, centerbias_tensor = preprocess_image(image_path)
        # Predict fixation using the model
        log_density_prediction = predict_fixation(image_tensor, centerbias_tensor)
        # Convert prediction to numpy for processing
        density_map = log_density_prediction.detach().cpu().numpy()[0, 0]

        # Create a figure for saving the image without displaying
        fig, ax = plt.subplots()
        ax.imshow(image)  # Plot original image
        # Overlay heatmap
        density_map_normalized = np.exp(density_map)  # Convert log density map to density map if needed

        intensity_threshold = np.percentile(density_map_normalized, 50)

        labeled_areas, num_features, total_area = calculate_highlighted_areas(density_map_normalized, intensity_threshold)

        # find at least 3 fixation points or twice the number of features (calculated from highlighted areas)
        # guideline: do not try to pass a number of points too high as it may not be possible to find that many points
        if max_points is not None:
            num_points_to_find = min(max(min_points, num_features * 2), max_points)
        else:
            num_points_to_find = max(min_points, num_features * 2)
       
        # norm = Normalize(vmin=np.min(density_map_normalized), vmax=np.max(density_map_normalized))
        # ax.imshow(density_map_normalized, cmap='inferno', alpha=0.7, extent=(0, image.shape[1], image.shape[0], 0))
        cmap_choice = color  # 'hot_r', 'inferno', 'magma', 'plasma' are good choices for heatmaps

        # lower percentile to highlight more areas
        # higher percentile to highlight fewer areas
        # higher second value = narrower range of values highlighted, less discritization
        # lower second value = higher range of values highlighted, more discritization
        # higher first value = higher bound of intesity, low intensity values will not be highlighted
        # lower first value = lower bound of intensity, more lower intensity values will be highlighted
        vmin, vmax = np.percentile(density_map_normalized, [5, 65])  # Adjust to highlight more areas
        norm = Normalize(vmin=vmin, vmax=vmax)

        # Overlay heatmap with adjustments
        ax.imshow(density_map_normalized, cmap=cmap_choice, alpha=0.7, norm=norm, extent=(0, image.shape[1], image.shape[0],0))
        # find how to increase the size of the fixation points
        if(show_fixation_points):
            fixation_points = find_fixation_points(density_map, num_points=num_points_to_find)  # Find fixation points
            intensities = range(len(fixation_points))  # Assuming intensities are needed for color mapping

            # Overlay fixation points with color mapping based on intensities
            # ax.scatter(fixation_points[:, 1], fixation_points[:, 0], c=intensities, cmap='inferno')

            fixation_point_size = 50 # Increase size for better visibility
            ax.scatter(fixation_points[:, 1], fixation_points[:, 0], c=intensities, cmap=cmap_choice, s=fixation_point_size)

            for i, (y, x) in enumerate(fixation_points):
                color = 'purple' if i == 0 else 'red'
                ax.scatter(x, y, c=color)  # Mark fixation point with color
                if i > 0:
                    prev_y, prev_x = fixation_points[i-1]
                    ax.plot([prev_x, x], [prev_y, y], color=color, linewidth=2)  # Draw line in red

        ax.axis('off')  # Optional: Remove axes for cleaner image

        # Save the plot to a BytesIO buffer
        buf = BytesIO()
        plt.savefig(buf, format='png', bbox_inches='tight', pad_inches=0)
        plt.close(fig)  # Close the figure to free memory
        buf.seek(0)

        # Convert buffer to base64 string
        base64_image = base64.b64encode(buf.getvalue()).decode('utf-8')
        buf.close()
        return base64_image

    except Exception as e:
        return {"status": 500, "message": f"Failed to generate heatmap: {str(e)}"}

    # def generate_base64_heatmap(image_path):
    # print('hello from predictiveEyeTracking')
    # # Preprocess the image and get tensors
    # image, image_tensor, centerbias_tensor = preprocess_image(image_path)
    # print('preprocessed image')
    # # Predict fixation using the model
    # log_density_prediction = predict_fixation(image_tensor, centerbias_tensor)
    # print('predicted fixation')
    # # Convert prediction to numpy for processing
    # density_map = log_density_prediction.detach().cpu().numpy()[0, 0]
    # print('converted prediction to numpy')
    # # Optionally, apply any overlays or modifications to the image here

    # # Create a figure for saving the image without displaying
    # fig, ax = plt.subplots()
    # print('created figure')
    # ax.imshow(image)  # Plot original image
    # print('plotted original image')
    # # Overlay heatmap
    # density_map_normalized = np.exp(density_map)  # Convert log density map to density map if needed
    # ax.imshow(density_map_normalized, cmap='jet', alpha=0.5, extent=(0, image.shape[1], image.shape[0], 0))

    # fixation_points = find_fixation_points(density_map)  # Find fixation points

    # intensities = range(len(fixation_points))

    # ax.scatter(fixation_points[:, 1], fixation_points[:, 0],c=intensities, cmap='viridis')  # Overlay fixation points
    # ax.axis('off')  # Optional: Remove axes for cleaner image

    # # Save the plot to a BytesIO buffer
    # buf = BytesIO()
    # plt.savefig(buf, format='png', bbox_inches='tight', pad_inches=0)
    # plt.close(fig)  # Close the figure to free memory
    # buf.seek(0)

    # # Convert buffer to base64 string
    # base64_image = base64.b64encode(buf.getvalue()).decode('utf-8')
    # buf.close()
    # return base64_image