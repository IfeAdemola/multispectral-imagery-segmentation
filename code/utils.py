import numpy as np
import matplotlib.pyplot as plt
from matplotlib import colors
import matplotlib.patches as mpatches
import torch
import os
import pickle
from matplotlib.colors import LinearSegmentedColormap

def get_file_paths(root_dir):
    """get the file path of all pickle files in a directory"""
    file_paths = [os.path.join(root_dir, file_name) for file_name in os.listdir(root_dir) if file_name.endswith(".pkl")]
    return file_paths

def get_filename_from_path(filepath):
    """Get file name excluding the extension from the file path """
    # Extract the base name (file name with extension)
    base_name = os.path.basename(filepath)
    
    # Split the base name to separate the file name and extension
    file_name, _ = os.path.splitext(base_name)
    
    return file_name

def load_array(file_path):
    with open(file_path, 'rb') as f:
        return pickle.load(f)
    
def convert_to_np(tensor):
    # convert pytorch tensors to numpy arrays
    if not isinstance(tensor, np.ndarray):
        tensor = tensor.cpu().numpy()
    return tensor

def display_input(array):
    """Have RGB channels of the image data in appropriate form"""
    image_to_plot = array[..., :3]

    # Plotting the image
    min_vals = np.min(image_to_plot, axis=(0, 1))
    max_vals = np.max(image_to_plot, axis=(0, 1))

    image_scaled = (image_to_plot - min_vals) / (max_vals - min_vals)
    image_scaled = np.clip(image_scaled, 0, 1)  # Clip values to [0, 1] range

    # Apply gamma correction to enhance contrast
    gamma = 0.4  # Adjust this value to control contrast
    image_corrected = np.power(image_scaled, gamma)

    # Adjust gain to control brightness
    gain = [1.2, 1.0, 0.8]  # Adjust gain for each channel (R, G, B)
    image_corrected = image_corrected * gain

    # Clip values to [0, 1] range
    image_corrected = np.clip(image_corrected, 0, 1)

    # Rearrange channels from BGR to RGB
    image_corrected_rgb = image_corrected[..., ::-1]
    return image_corrected_rgb

def display_label(array, label_structure):
    cmap = mycmap(label_structure)

    if len(cmap.colors) == 3:
        norm = colors.Normalize(vmin=0, vmax=2) # not sure which works better cuz some images have just 2 labels
    elif len(cmap.colors) == 6:
        norm = colors.Normalize(vmin=0, vmax=5)
    array = cmap(norm(array))
    array_uint8 = (array * 255).astype(np.uint8)

    return array_uint8

def plot_image(array):
    """Plot RGB channels of the image data"""
    image_to_plot = array[..., :3]

    # Plotting the image
    min_vals = np.min(image_to_plot, axis=(0, 1))
    max_vals = np.max(image_to_plot, axis=(0, 1))

    image_scaled = (image_to_plot - min_vals) / (max_vals - min_vals)
    image_scaled = np.clip(image_scaled, 0, 1)  # Clip values to [0, 1] range

    # Apply gamma correction to enhance contrast
    gamma = 0.4  # Adjust this value to control contrast
    image_corrected = np.power(image_scaled, gamma)

    # Adjust gain to control brightness
    gain = [1.2, 1.0, 0.8]  # Adjust gain for each channel (R, G, B)
    image_corrected = image_corrected * gain

    # Clip values to [0, 1] range
    image_corrected = np.clip(image_corrected, 0, 1)

    # Rearrange channels from BGR to RGB
    image_corrected_rgb = image_corrected[..., ::-1]
    image_corr = image_corrected_rgb[100:200, 100:200]
    # Plotting the corrected image
    plt.figure(figsize=(8,8)) 
    plt.imshow(image_corrected_rgb)
    plt.axis('off')  # Turn off axis
    plt.show()

def plot_channel(array, band_idx=0):
    """
    Plot a specific color band of a multispectral image.
    
    Parameters:
    - array (numpy.ndarray): Multispectral image array.
    - band_idx (int): Index of the color band to plot (0 for Red, 1 for Green, 2 for Blue).
    """
    image_to_plot = array[..., band_idx]

    # Normalize the image
    min_val = np.min(image_to_plot)
    max_val = np.max(image_to_plot)
    image_normalized = (image_to_plot - min_val) / (max_val - min_val)
    
    # Apply gamma correction to enhance contrast
    gamma = 0.4  # Adjust this value to control contrast
    image_corrected = np.power(image_normalized, gamma)

    # Clip values to [0, 1] range
    image_corrected = np.clip(image_corrected, 0, 1)

    colors = ['#CCCCCC', '#D2B48C', '#228B22'] 
    # Create a colormap with three colors
    cmap_name = 'custom_cmap'
    custom_cmap = LinearSegmentedColormap.from_list(cmap_name, colors, N=3)

    # Plotting the corrected image
    plt.figure(figsize=(10,10))
    plt.imshow(image_corrected, cmap='gray')  # Use custom colormap for single band image
    plt.axis('off')  # Turn off axis
    plt.colorbar(ticks = [0,1,2])  # Add color bar to show intensity values
    plt.title(f'Band {band_idx + 1}')  # Add title with band index
    plt.show()

def plot_label(array):
    cmap = mycmap()
    # vmin = np.min(array)
    # vmax = np.max(array)
    norm = colors.Normalize(vmin=0, vmax=2) # not sure which works better cuz some images have just 2 labels
    array = cmap(norm(array))
    array_uint8 = (array * 255).astype(np.uint8)

    plt.figure(figsize=(10,10))
    plt.imshow(array, cmap=cmap, interpolation='nearest')  # Use custom colormap for single band image
    plt.axis('off')  # Turn off axis
    # plt.colorbar(ticks = [0,1,2])  # Add color bar to show intensity values
    # plt.title(f'Band {12 + 1}')  # Add title with band index
    plt.show()


# def plot_label(label):
#     """plot label channel from preprocessed data or patches, that is, groundtruth"""
#     colors = ['#CCCCCC', '#D2B48C', '#228B22'] 
#     # Create a colormap with three colors
#     cmap_name = 'custom_cmap'
#     custom_cmap = LinearSegmentedColormap.from_list(cmap_name, colors, N=3)

#     plt.figure(figsize=(10,10))
#     plt.imshow(label, cmap=custom_cmap, interpolation='nearest')  # Use custom colormap for single band image
#     plt.axis('off')  # Turn off axis
#     plt.colorbar(ticks = [0,1,2])  # Add color bar to show intensity values
#     plt.title(f'Band {12 + 1}')  # Add title with band index
#     plt.show()

# def mycmap():
#     cmap = colors.ListedColormap(["#CCCCCC",
#                                   "#D2B48C",
#                                   "#228B22"
#                                 ])
#     return cmap

def classnames(label_structure=False):
    if label_structure:
        names = ["Invalid", "Soil", "Low grass", "High grass", "Partial trees", "Forest"]
    else:
        names = ["Invalid", "Deforested", "Forest"]
    return names

def labels(label_structure):
    l = {}
    for i, label in enumerate(classnames(label_structure)):
        l[i] = label
        return l

def mycmap(label_structure=False):
    if label_structure:
        cmap = colors.ListedColormap(["#CCCCCC",
                                    "#D2B48C",
                                    "#90EE90",
                                    "#006400",
                                    "#556B2F",
                                    "#228B22"])  # ,"#FFFFFF"
    else:
        cmap = colors.ListedColormap(["#CCCCCC",
                                      "#D2B48C",
                                      "#228B22"])
    return cmap

def mypatches():
    patches = []
    for counter, name in enumerate(classnames(label=False)):
        patches.append(mpatches.Patch(color=mycmap().colors[counter],
                                      label=name))
    return patches