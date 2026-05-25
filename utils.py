import os
import re
import torch
import numpy as np
from tqdm import tqdm
from torch.utils.data import DataLoader
from sklearn.model_selection import train_test_split
from dataset import CellDataset
import tifffile
from skimage.segmentation import find_boundaries


def extract_instance_boundaries(segmentation_map):
    instance_boundaries = []
    for instance_label in np.unique(segmentation_map):
        if instance_label == 0:  # Skip the background
            continue
        instance_mask = segmentation_map == instance_label
        boundaries = find_boundaries(np.array(instance_mask).astype(np.int16), mode='outer')
        points = np.column_stack(np.where(boundaries))
        instance_boundaries.append(points)
    return instance_boundaries


def compute_mean_std(train_dataloader, stop=30):
    pixel_values = []

    for i, (_, image, _) in enumerate(tqdm(train_dataloader)):
        if i == stop:
            break
        pixel_values.append(torch.ravel(image))

    pixels = np.concatenate(pixel_values)
    mean = np.mean(pixels)
    std = np.std(pixels)

    return mean, std

def print_dir_tree(start_path, indent=0):
    """Prints the tree of directories starting from the given directory path."""
    if indent == 0:
        print(start_path)

    for item in os.listdir(start_path):
        path = os.path.join(start_path, item)
        if os.path.isdir(path):
            print('    ' * indent + '|-- ' + item)
            print_dir_tree(path, indent + 1)



def get_images_and_masks(name, path):
    """
    Returns a sorted path list of both images_dir and labels_dir
    """
    images_path = os.path.abspath(f'{path}/{name}')
    labels_path = os.path.abspath(f'{path}/{name}_GT/SEG')

    # Get full paths and sort them based on the numerical part of the filenames
    images_dir = sorted(
        [os.path.join(images_path, f) for f in os.listdir(images_path)],
        key=lambda x: int(re.search(r'\d+', os.path.basename(x)).group())
    )
    labels_dir = sorted(
        [os.path.join(labels_path, f) for f in os.listdir(labels_path)],
        key=lambda x: int(re.search(r'\d+', os.path.basename(x)).group())
    )

    return images_dir, labels_dir



def separate_masks(instance_mask):
    unique_labels = np.unique(instance_mask)
    unique_labels = unique_labels[unique_labels != 0]  # remove background label

    binary_masks = []

    for label in unique_labels:
        binary_mask = np.zeros_like(instance_mask)
        binary_mask[instance_mask == label] = 1
        binary_masks.append(binary_mask)

    return binary_masks


def generate_train_test_data(valid_seg_paths, valid_images_paths, output_path, is_trinary, transform, metric, num_workers=8, num_epochs=1, is_3d=False, dilation_iterations=0):
    valid_seg_paths, test_seg_paths, valid_images_paths, test_images_paths = train_test_split(valid_seg_paths,
                                                                                              valid_images_paths,
                                                                                              test_size=0.5,
                                                                                              shuffle=True,
                                                                                              random_state=42
                                                                                             )

    valid_dataset = CellDataset(valid_seg_paths, valid_images_paths, metric,
                                transform, is_trinary=is_trinary, normalize=False,
                                is_3d=is_3d, dilation_iterations=dilation_iterations)
    test_dataset = CellDataset(test_seg_paths, test_images_paths, metric,
                               transform, is_trinary=is_trinary,
                               normalize=False, is_3d=is_3d, dilation_iterations=dilation_iterations)
    valid_dataloader = DataLoader(valid_dataset, batch_size=16, shuffle=True, num_workers=num_workers)
    test_dataloader = DataLoader(test_dataset, batch_size=16, shuffle=True, num_workers=num_workers)

    # Process each dataloader
    for mode, loader in [("valid", valid_dataloader), ("test", test_dataloader)]:
        image_dir = os.path.join(output_path, mode, 'images')
        seg_dir = os.path.join(output_path, mode, 'segmentations')
        label_dir = os.path.join(output_path, mode, 'labels')

        # Create directories if they do not exist
        os.makedirs(image_dir, exist_ok=True)
        os.makedirs(seg_dir, exist_ok=True)
        os.makedirs(label_dir, exist_ok=True)

        for epoch in range(num_epochs): # generate more test/valid data incase of datasets with fewer examples
            for i, (images, transformed_segs, labels, _) in enumerate(loader):
                for j, (image, seg, label) in enumerate(zip(images, transformed_segs, labels)):
                    image_path = os.path.join(image_dir, f"{epoch}_{i}_{j}.tiff")
                    seg_path = os.path.join(seg_dir, f"{epoch}_{i}_{j}.tiff")
                    label_path = os.path.join(label_dir, f"{epoch}_{i}_{j}.txt")

                    image_np = image.cpu().detach().numpy()
                    seg_np = seg.cpu().detach().numpy()

                    image_np = image_np.astype(np.float64)
                    seg_np = seg_np.astype(np.float64)

                    tifffile.imwrite(image_path, image_np)
                    tifffile.imwrite(seg_path, seg_np)

                    with open(label_path, 'w') as f:
                        f.write(f"{label.item()}")


def list_dataset_files(base_path):
    images = []
    segmentations = []
    seg_labels = []

    # Define the subdirectories
    subdirs = {
        "images": images,
        "segmentations": segmentations,
        "labels": seg_labels
    }

    # Walk through the base directory and collect files
    for subdir, file_list in subdirs.items():
        dir_path = os.path.join(base_path, subdir)

        # Check if directory exists
        if os.path.exists(dir_path):
            # List all files in the directory
            for file in os.listdir(dir_path):
                full_path = os.path.join(dir_path, file)
                file_list.append(full_path)
        else:
            print(f"Directory does not exist: {dir_path}")

    return images, segmentations, seg_labels
