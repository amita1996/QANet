from utils import *
import torch
import numpy as np
from scipy.spatial import cKDTree


class Metrics:
    def __init__(self, metric, percentile=100):
        self.metric_name = metric
        self.percentile = percentile
        if metric == 'ASD':
            self.metric = self.average_surface_distance
        elif metric == 'hausdorff':
            self.metric = self.hausdorff_distance
        elif metric == 'seg':
            self.metric = self.calc_SEG_measure
        elif metric == '3d_seg':
            self.metric = self.calc_SEG_measure_3d
        else:
            raise ValueError(f"Unsupported metric: {metric}")

    def __call__(self, predicted_map, ground_truth_map):
        if self.metric_name in ['ASD', 'hausdorff']:
            return self.calc_instance_distance_metric(predicted_map, ground_truth_map)
        elif self.metric_name == 'seg':
            return self.calc_SEG_measure(predicted_map, ground_truth_map)
        elif self.metric_name == '3d_seg':
            return self.calc_SEG_measure_3d(predicted_map, ground_truth_map)
        else:
            raise ValueError(f"Unsupported metric: {self.metric_name}")

    def calc_instance_distance_metric(self, predicted_map, ground_truth_map):
        """
        Calculates the specified distance metric (e.g., Hausdorff, ASD) for the entire segmentation map.
        """
        pred_instances = extract_instance_boundaries(predicted_map)
        gt_instances = extract_instance_boundaries(ground_truth_map)

        distances = []
        for gt_idx in range(len(gt_instances)):
            curr_distances = [self.metric(pred_boundary, gt_instances[gt_idx]) for pred_boundary in pred_instances]

            if curr_distances:
                distances.append(min(curr_distances))
            else:
                distances.append(np.inf)  # Assign a large value if no predicted instance is found

        if len(distances) == 0:
            return 0.0, distances

        distances = np.array(distances)
        return np.mean(distances), distances

    def calc_SEG_measure(self, pred_labels_mask, gt_labels_mask):
        binary_masks_predicted = separate_masks(pred_labels_mask)
        binary_masks_gt = separate_masks(gt_labels_mask)

        SEG_measure_array = np.zeros(len(binary_masks_gt))
        for i, r in enumerate(binary_masks_gt):
            r_and_s = s = None
            # Find a matching predicted mask
            for s in binary_masks_predicted:
                r_and_s = np.logical_and(r, s)
                if np.sum(r_and_s) > 0.5 * np.sum(r):
                    # Match found
                    break

            # Calculate Jaccard similarity index
            if r_and_s is not None and s is not None:
                j_similarity = np.sum(r_and_s) / np.sum(np.logical_or(r, s))
            else:
                j_similarity = 0

            SEG_measure_array[i] = j_similarity

        SEG_measure_avg = np.average(SEG_measure_array) if len(SEG_measure_array) > 0 else 0
        return SEG_measure_avg, SEG_measure_array

    def calc_SEG_measure_3d(self, pred_labels_mask, gt_labels_mask):
        # Flatten the masks and remove the background (0) label
        pred_labels_mask = pred_labels_mask.flatten()
        gt_labels_mask = gt_labels_mask.flatten()

        # Get unique labels excluding the background
        pred_unique_labels = torch.unique(pred_labels_mask)
        gt_unique_labels = torch.unique(gt_labels_mask)

        # Remove the background label (assumed to be 0)
        pred_unique_labels = pred_unique_labels[pred_unique_labels != 0]
        gt_unique_labels = gt_unique_labels[gt_unique_labels != 0]

        # Initialize SEG measure array
        SEG_measure_array = torch.zeros(len(gt_unique_labels), device=pred_labels_mask.device)

        # Iterate over each unique ground truth label
        for i, gt_label in enumerate(gt_unique_labels):
            gt_mask = (gt_labels_mask == gt_label)
            gt_size = gt_mask.sum().item()

            # Find matching predicted label based on Jaccard similarity
            max_j_similarity = 0
            for pred_label in pred_unique_labels:
                pred_mask = (pred_labels_mask == pred_label)
                r_and_s = (gt_mask & pred_mask).sum().item()

                if r_and_s > 0.5 * gt_size:
                    j_similarity = r_and_s / (gt_size + pred_mask.sum().item() - r_and_s)
                    max_j_similarity = max(max_j_similarity, j_similarity)

            SEG_measure_array[i] = max_j_similarity

        SEG_measure_avg = SEG_measure_array.mean()
        return SEG_measure_avg, SEG_measure_array

    def get_directed_distances(self, u, v):
        a_points = np.array(u)
        b_points = np.array(v)

        if len(a_points) == 0:
            return np.inf
        elif len(b_points) == 0:
            return np.inf

        fwd, bwd = (
            cKDTree(a_points).query(b_points, k=1)[0],
            cKDTree(b_points).query(a_points, k=1)[0],
        )

        return fwd, bwd

    def average_surface_distance(self, u, v):
        min_distances_u_to_v, min_distances_v_to_u = self.get_directed_distances(u, v)

        sum_distances = np.sum(min_distances_u_to_v) + np.sum(min_distances_v_to_u)
        normalizing_factor = 1 / (len(min_distances_u_to_v) + len(min_distances_v_to_u))
        asd = normalizing_factor * sum_distances

        return asd

    def hausdorff_distance(self, u, v):
        fwd, bwd = self.get_directed_distances(u, v)

        if self.percentile < 100:
            threshold_fwd = np.percentile(fwd, self.percentile)
            threshold_bwd = np.percentile(bwd, self.percentile)
            fwd = fwd[fwd <= threshold_fwd]
            bwd = bwd[bwd <= threshold_bwd]

        # Calculate the directed Hausdorff distances
        directed_hausdorff_u_to_v = np.mean(fwd)
        directed_hausdorff_v_to_u = np.mean(bwd)

        # Calculate the Modified Hausdorff Distance
        mhd = max(directed_hausdorff_u_to_v, directed_hausdorff_v_to_u)

        return float(mhd)