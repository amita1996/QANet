from scipy.ndimage import grey_dilation, grey_erosion, grey_closing, grey_opening, binary_dilation
from scipy.ndimage.morphology import generate_binary_structure
from torch.utils.data import Dataset
import torch
from skimage.segmentation import find_boundaries
from PIL import Image
import tifffile
import random
import numpy as np
import cv2
from scipy.ndimage import gaussian_filter
from torchvision.transforms.v2 import ElasticTransform
from torchvision.transforms import InterpolationMode
import torchio as tio
from scipy import ndimage


class CellDataset(Dataset):
    def __init__(self, seg_dir, image_dir, metric,
                 transform=None, train=True, is_trinary=True, label_dir=None, normalize=True, is_3d=False,
                 dilation_iterations=0):
        """
        Initialize the CellDataset.

        Parameters
        ----------
        seg_dir : list[str]
            Paths to ground truth segmentation files (TIFF).
        image_dir : list[str]
            Paths to microscopy image files (TIFF).
        metric : callable
            Function to compute metrics between ground truth and transformed segmentation.
        transform : callable, optional
            Optional transform (e.g., elastic) to apply to image and mask.
        train : bool, default=True
            Whether the dataset is in training mode (applies augmentations).
        is_trinary : bool, default=True
            If True, use trinary labels (background, cell, boundary), else binary (background, cell).
        label_dir : list[str], optional
            Paths to labels (e.g., numeric metrics) for validation mode.
        normalize : bool, default=True
            If True, apply min-max normalization to image and segmentation.
        is_3d : bool, default=False
            If True, treat data as 3D volumes; else 2D slices.
        dilation_iterations : int, default=0
            Number of iterations to dilate boundary maps for trinary labels.
        """

        self.seg_dir = seg_dir
        self.image_dir = image_dir
        self.metric = metric
        self.transform = transform
        self.train = train
        self.is_trinary = is_trinary

        # min max scaling - normalizing to get values between 0 and 1 for
        # both the segmentation and the image

        self.image_mean, self.image_std = (0, 255)
        self.seg_mean, self.seg_std = (0, 2 if self.is_trinary else 1)

        self.label_dir = label_dir
        self.normalize = normalize
        self.is_3d = is_3d

        # determines how big the border will be. 3D images need bigger border
        self.dilation_iterations = dilation_iterations

    def __len__(self):
        return len(self.seg_dir)

    def erode_dilate_close_open(self, labeled_seg):
        """
        Apply random morphological operations (erosion, dilation, opening, closing)
        to each labeled cell in the segmentation map.

        Parameters
        ----------
        labeled_seg : np.ndarray
            Integer array of shape (Z, H, W) or (H, W) with labeled regions.

        Returns
        -------
        np.ndarray
            Modified segmentation with the same shape and labels.
        """

        labeled_seg_out = np.zeros_like(labeled_seg)
        labels, num_labels = ndimage.label(labeled_seg)
        for label in range(1, num_labels + 1):
            cell_mask = labels == label
            cell_size = np.sum(cell_mask)
            # Adjust structuring element size based on cell size
            if self.is_3d:
                if cell_size < 1200:
                    size = random.choice(range(1, 4))
                    depth_size = random.choice(range(1, 4))
                else:
                    size = random.choice(range(3, 15))
                    depth_size = random.choice(range(2, 10))
                strel = np.ones((depth_size, size, size))

            else:
                if cell_size < 2000:
                    size = random.choice(range(5, 20))
                else:
                    size = random.choice(range(10, 25))
                strel = np.ones((size, size))
            ed = np.random.rand() < 0.5
            if np.random.rand() < 0.5:
                if ed:
                    cell_transformed = grey_dilation(cell_mask.astype(np.uint8), footprint=strel)
                else:
                    cell_transformed = grey_opening(cell_mask.astype(np.uint8), footprint=strel)
            else:
                if ed:
                    cell_transformed = grey_erosion(cell_mask.astype(np.uint8), footprint=strel)
                else:
                    cell_transformed = grey_closing(cell_mask.astype(np.uint8), footprint=strel)
            labeled_seg_out[cell_transformed > 0] = label
        return labeled_seg_out

    @staticmethod
    def create_vector_field(size, low=-512, high=512, sigma=38):
        """
        Create a random vector field and smooth it using a Gaussian kernel.
        """
        vx = np.random.uniform(low, high, size)
        vy = np.random.uniform(low, high, size)

        vx_smooth = gaussian_filter(vx, sigma)
        vy_smooth = gaussian_filter(vy, sigma)

        return vx_smooth, vy_smooth

    @staticmethod
    def apply_vector_field(segmentation, vx, vy):
        """
        Apply the vector field to the segmentation.
        """
        h, w = segmentation.shape
        x, y = np.meshgrid(np.arange(w), np.arange(h))

        map_x = (x + vx).astype(np.float32)
        map_y = (y + vy).astype(np.float32)

        deformed_segmentation = cv2.remap(segmentation, map_x, map_y, interpolation=cv2.INTER_NEAREST,
                                          borderMode=cv2.BORDER_REFLECT, borderValue=0)

        return deformed_segmentation

    def synthesize_imperfect_segmentation(self, gt_segmentation, morphology_state):
        """
        Synthesize imperfect segmentations based on the morphology state.
        """

        if self.is_3d:
            if morphology_state:
                num_control_points = (random.randint(5, 8), random.randint(5, 8), random.randint(5, 8))
                max_displacement = (random.uniform(0, 1), random.uniform(2, 4), random.uniform(2, 4))

                # Apply the random elastic transformation
                elastic_transform = tio.RandomElasticDeformation(
                    num_control_points=num_control_points,  # Randomized control points
                    max_displacement=max_displacement,  # Randomized displacement
                    label_interpolation='nearest',
                    locked_borders=2  # Keep the locked borders
                )

                subject = tio.Subject(
                    mask=tio.LabelMap(tensor=gt_segmentation[np.newaxis, :, :, :])
                )
                deformed_segmentation = np.array(elastic_transform(subject)['mask'].data[0])
            else:
                deformed_segmentation = gt_segmentation

        else:
            if morphology_state:
                vx, vy = self.create_vector_field(gt_segmentation.shape)
                deformed_segmentation = self.apply_vector_field(gt_segmentation, vx, vy)

            else:
                deformed_segmentation = gt_segmentation

        return deformed_segmentation

    def transform_seg(self, seg, nonrigid_p=0.2):
        """
        Apply morphological operations and non-rigid deformation to a segmentation.

        Parameters
        ----------
        seg : np.ndarray
            Input ground truth segmentation map.
        nonrigid_p : float, default=0.2
            Probability threshold for applying non-rigid deformation.

        Returns
        -------
        torch.Tensor
            Transformed segmentation as a Tensor.
        """

        seg = self.erode_dilate_close_open(seg)
        seg = self.synthesize_imperfect_segmentation(seg, int(nonrigid_p < torch.rand(1)))
        seg = self.merge_labels(seg)

        if not torch.is_tensor(seg):
            seg = torch.tensor(seg)

        return seg

    def label_segmentation_map(self, mask):
        """
        Function to map a ground truth segmentation map to a segmentation map with 3 classes:
        0 -> Background
        1 -> Cell
        2 -> Boundaries
        """
        for curr_mask in mask:

            if self.is_trinary:
                # Find boundaries
                boundary_map = find_boundaries(np.array(curr_mask).astype(np.int16), mode='outer')

                if self.dilation_iterations > 0:
                    boundary_map = binary_dilation(boundary_map, iterations=self.dilation_iterations)

                # Set non-background pixels to 1 (cells)
                curr_mask[curr_mask != 0] = 1

                # Set boundary pixels to 2
                curr_mask[boundary_map != 0] = 2

            else:
                # Set non-background pixels to 1 (cells)
                curr_mask[curr_mask != 0] = 1

        return mask

    def merge_labels(self, labeled_seg, merge_prob=0.3):
        """
        Randomly merge adjacent segmentation labels to simulate over-segmentation merging.

        Parameters
        ----------
        labeled_seg : np.ndarray
            Input labeled segmentation map.
        merge_prob : float, default=0.3
            Probability of merging each adjacent label pair.

        Returns
        -------
        np.ndarray
            Segmentation map with some labels merged.
        """

        strel = generate_binary_structure(3 if self.is_3d else 2, 1)
        labeled_seg_dilate = grey_dilation(labeled_seg, footprint=strel)
        diff = abs(labeled_seg_dilate - labeled_seg) > 0
        diff = np.logical_and(diff, labeled_seg > 0)
        diff = np.logical_and(diff, labeled_seg_dilate > 0)
        orig_labels = labeled_seg[diff]
        dilated_labels = labeled_seg_dilate[diff]
        pairs = set(zip(orig_labels.ravel(), dilated_labels.ravel()))
        out_labels = labeled_seg.copy()
        for l1, l2 in pairs:
            if np.random.rand() < merge_prob:
                l1_out = out_labels[labeled_seg == l1].min()
                l2_out = out_labels[labeled_seg == l2].min()
                min_l = np.minimum(l1_out, l2_out)
                max_l = np.maximum(l1_out, l2_out)

                out_labels[out_labels == max_l] = min_l
        return out_labels

    def __getitem__(self, idx):
        if not self.train:
            transformed_seg = np.array(tifffile.imread(self.seg_dir[idx])).astype(np.float64)
            image = np.array(tifffile.imread(self.image_dir[idx])).astype(np.float64)
            label = np.loadtxt(self.label_dir[idx])

            # instead of generating the same data twice, we generate only trinary data. and then if we are running
            # a network with binary data, if change the trinary data to binary
            if not self.is_trinary:
                for curr_mask in transformed_seg:
                    curr_mask[curr_mask != 0] = 1

            if self.normalize:
                image = (image - self.image_mean) / self.image_std
                transformed_seg = (transformed_seg - self.seg_mean) / self.seg_std

            return image, transformed_seg, label, torch.tensor([0])

        gt_seg = tifffile.imread(self.seg_dir[idx]).astype(np.float64)
        image = tifffile.imread(self.image_dir[idx]).astype(np.float64)

        if self.transform:
            if self.is_3d:
                subject = tio.Subject(
                    image=tio.ScalarImage(tensor=image[np.newaxis, :, :, :]),
                    mask=tio.LabelMap(tensor=gt_seg[np.newaxis, :, :, :])
                )

                out = self.transform(subject)
                image = out['image'].data[0]
                gt_seg = out['mask'].data[0]

            else:
                out = self.transform(image=image, mask=gt_seg)
                gt_seg = out['mask']
                image = out['image']

        transformed_seg = self.transform_seg(gt_seg)

        if not torch.is_tensor(gt_seg):
            gt_seg = torch.tensor(gt_seg)

        label, _ = self.metric(gt_seg, transformed_seg)
        transformed_seg = self.label_segmentation_map(transformed_seg)
        gt_seg = self.label_segmentation_map(gt_seg)

        if image.max() > 256:
            image = (255 / 65535) * image  # for Cytopacq images.

        if self.normalize:
            image = (image - self.image_mean) / self.image_std
            transformed_seg = (transformed_seg - self.seg_mean) / self.seg_std

        if not torch.is_tensor(label):
            label = torch.tensor(label)

        return image, transformed_seg, label, gt_seg
