from scipy.ndimage import grey_dilation, grey_erosion, grey_closing, grey_opening
from scipy.ndimage.morphology import generate_binary_structure
from scipy.ndimage.interpolation import map_coordinates
from torch.utils.data import Dataset
import torch
from skimage.segmentation import find_boundaries
from PIL import Image
import tifffile
import random
import numpy as np
import cv2
from scipy.ndimage import gaussian_filter


class CellDataset(Dataset):
    def __init__(self, seg_dir, image_dir, metric, transform=None, train=True, is_trinary=True, label_dir=None, normalize=True):
        self.seg_dir = seg_dir
        self.image_dir = image_dir
        self.metric = metric
        self.transform = transform
        self.train = train
        self.is_trinary = is_trinary
        self.image_mean, self.image_std = (96.45106709798178, 13.568400191066292)
        self.seg_mean, self.seg_std = (0.14350941975911458, 0.37072217808206764)
        self.label_dir = label_dir
        self.normalize = normalize


    def __len__(self):
        return len(self.seg_dir)


    @staticmethod
    def _get_indices4elastic_transform(shape, alpha):
        f = 32
        shape = (int(shape[0] / f), int(shape[1] / f))
        dxr = np.random.rand(*shape)
        dx = cv2.resize((dxr * 2 - 1) * alpha, (0, 0), fx=f, fy=f)
        dy = cv2.resize((np.random.rand(*shape) * 2 - 1) * alpha, (0, 0), fx=f, fy=f)
        x, y = np.meshgrid(np.arange(shape[1] * f), np.arange(shape[0] * f))
        indices = np.reshape(y + dy, (-1, 1)), np.reshape(x + dx, (-1, 1))
        return indices


    @staticmethod
    def _get_transformed_image_(image, indices, seg=True):
        image_size = image.shape

        if seg:
            trans_coord = map_coordinates(image, indices, order=0, mode='reflect', cval=0).reshape(image_size)
        else:
            trans_coord = map_coordinates(image, indices, order=1, mode='reflect').reshape(image_size)

        return trans_coord


    @staticmethod
    def erode_dilate_merge(labeled_seg):
        ed = np.random.rand(1) < 0.5
        size = random.choice([7, 9, 11, 13])
        strel = np.ones((size, size))

        if np.random.rand(1) < 0.5:
            if ed:
                labeled_seg_out = grey_dilation(labeled_seg, footprint=strel)
            else:
                labeled_seg_out = grey_opening(labeled_seg, footprint=strel)
        else:
            if ed:
                labeled_seg_out = grey_erosion(labeled_seg, footprint=strel)
            else:
                labeled_seg_out = grey_closing(labeled_seg, footprint=strel)

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
        if morphology_state != 0:
            sigma_mo = np.random.randint(1, 5)  # U(1, 4)
            kernel_size = (sigma_mo, sigma_mo)
            gt_segmentation = cv2.morphologyEx(gt_segmentation, cv2.MORPH_OPEN, np.ones(kernel_size, np.uint8))

        vx, vy = self.create_vector_field(gt_segmentation.shape)
        deformed_segmentation = self.apply_vector_field(gt_segmentation, vx, vy)

        return deformed_segmentation

    def transform_seg(self, seg, nonrigid_p=0.2):
        seg = self.erode_dilate_merge(seg)
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

                # Set non-background pixels to 1 (cells)
                curr_mask[curr_mask != 0] = 1

                # Set boundary pixels to 2
                curr_mask[boundary_map != 0] = 2

            else:
                # Set non-background pixels to 1 (cells)
                curr_mask[curr_mask != 0] = 1

        return mask


    @staticmethod
    def merge_labels(labeled_seg, merge_prob=0.4):
        strel = generate_binary_structure(2, 1)
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


    @staticmethod
    def _adjust_brightness_(image, delta):
        out_img = image + delta
        return out_img


    @staticmethod
    def _adjust_contrast_(image, factor):
        img_mean = image.mean()
        out_img = (image - img_mean) * factor + img_mean
        return out_img



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



        gt_seg = np.array(Image.open(self.seg_dir[idx])).astype(np.float64)
        image = np.array(Image.open(self.image_dir[idx])).astype(np.float64)


        if self.transform:
            out = self.transform(image=image, mask=gt_seg)
            gt_seg = out['mask']
            image = out['image']

        transformed_seg = self.transform_seg(gt_seg)
        gt_seg = torch.tensor(gt_seg)

        label, _ = self.metric(gt_seg, transformed_seg)
        transformed_seg = self.label_segmentation_map(transformed_seg)
        gt_seg = self.label_segmentation_map(gt_seg)

        image = (255 / 65535) * image # for Cytopacq images.

        if self.normalize:
            image = (image - self.image_mean) / self.image_std
            transformed_seg = (transformed_seg - self.seg_mean) / self.seg_std

        if not torch.is_tensor(label):
            label = torch.tensor(label)


        return image, transformed_seg, label, gt_seg
