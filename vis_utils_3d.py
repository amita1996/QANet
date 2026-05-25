import matplotlib.pyplot as plt
import matplotlib.cm as cm
from scipy.ndimage import zoom
import numpy as np
import plotly.graph_objects as go

# Function to visualize a 3D image slice
def plot_image_slice(tensor, slice_idx, title):
    plt.imshow(tensor[slice_idx, :, :].numpy(), cmap='gray')
    plt.axis('off')
    plt.title(title)


def plot_crop_image_slices(image, mask, cropped_image_tensor, cropped_mask_tensor, slice_idx):
    # Visualize a slice from the original image and mask
    plt.figure(figsize=(10, 10))
    plt.subplot(2, 2, 1)
    plot_image_slice(image, slice_idx=slice_idx, title='Original Image')  # Example slice index
    plt.subplot(2, 2, 2)
    plot_image_slice(mask, slice_idx=slice_idx, title='Original Mask')  # Example slice index

    # Visualize a slice from the cropped image and mask
    plt.subplot(2, 2, 3)
    plot_image_slice(cropped_image_tensor, slice_idx=slice_idx, title='Cropped Image')  # Example slice index
    plt.subplot(2, 2, 4)
    plot_image_slice(cropped_mask_tensor, slice_idx=slice_idx, title='Cropped Mask')  # Example slice index

    plt.show()


def plot3d_image_gt_pred_slices(image, gt_mask, pred_mask, num_slices, cmap='viridis', plt_title=None):
    # show first image in batch
    plt.figure(figsize=(12, 6))
    for i in range(num_slices):
        idx_show = i * len(image) // num_slices
        plt.subplot(3, num_slices, i + 1)
        plt.imshow(image[idx_show], cmap='gray')
        plt.title(f'Image[{idx_show}] ')

        plt.subplot(3, num_slices, i + num_slices + 1)
        plt.imshow(gt_mask[idx_show], cmap=cmap)
        plt.title(f'GT Mask[{idx_show}] ')

        plt.subplot(3, num_slices, i + 2*num_slices + 1)
        plt.imshow(pred_mask[idx_show], cmap=cmap)
        plt.title(f'Pred Mask[{idx_show}] ')

    if plt_title is not None:
        plt.suptitle(plt_title)
    plt.tight_layout()
    plt.show()


def save_3d_mask_plot(mask, save_path, zoom_scale=0.5):
    if zoom_scale is not None:
        mask = zoom(mask, (zoom_scale, zoom_scale, zoom_scale), order=0)

    # Get unique classes in the segmentation map
    classes = np.unique(mask)

    # Create a colormap with enough colors for all classes
    colors = cm.rainbow(np.linspace(0, 1, len(classes)))

    # Create a dictionary to map each class to a color
    color_map = {class_label: colors[i] for i, class_label in enumerate(classes)}

    # Create a figure to hold all scatter plots
    fig = go.Figure()

    # Plot each class separately
    for class_label in classes:
        if class_label == 0:
            continue
        # Extract coordinates of points belonging to the current class
        points = np.argwhere(mask == class_label)

        # Separate coordinates into x, y, z arrays
        x_coords = points[:, 1]
        y_coords = points[:, 2]
        z_coords = points[:, 0]

        # Convert RGB values to hex format
        rgb_color = tuple(int(color * 255) for color in color_map[class_label])
        hex_color = '#{0:02x}{1:02x}{2:02x}'.format(*rgb_color)

        # Create a scatter plot for the current class
        fig.add_trace(go.Scatter3d(
            x=x_coords,
            y=y_coords,
            z=z_coords,
            mode='markers',
            marker=dict(
                size=5,
                color=hex_color,
                opacity=0.8
            ),
            name=f'Class {class_label}'
        ))

    # Set layout
    fig.update_layout(scene=dict(
        xaxis_title='X',
        yaxis_title='Y',
        zaxis_title='Z'),
        width=700,
        margin=dict(r=20, l=10,
                    b=10, t=10))

    # Save the plot
    fig.write_html(save_path, auto_play=False)