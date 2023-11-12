from logging import raiseExceptions
import os.path as osp
import warnings
from collections import OrderedDict
from functools import reduce

import mmcv
import numpy as np
from mmcv.utils import print_log
from prettytable import PrettyTable
from torch.utils.data import Dataset

from depth.core import pre_eval_to_metrics, metrics, eval_metrics
from depth.utils import get_root_logger
from depth.datasets.builder import DATASETS
from depth.datasets.pipelines import Compose

from depth.ops import resize

from PIL import Image
import cv2

import torch


@DATASETS.register_module()
class WaymoDataset(Dataset):
    """Waymo dataset for depth estimation.
    split file format:
    input_image: 2011_09_26/2011_09_26_drive_0002_sync/image_02/data/0000000069.png 
    gt_depth:    2011_09_26_drive_0002_sync/proj_depth/groundtruth/image_02/0000000069.png 
    focal:       721.5377 (following the focal setting in BTS, but actually we do not use it)
    Args:
        pipeline (list[dict]): Processing pipeline
        img_dir (str): Path to image directory
        ann_dir (str, optional): Path to annotation directory. Default: None
        split (str, optional): Split txt file. Split should be specified, only file in the splits will be loaded.
        data_root (str, optional): Data root for img_dir/ann_dir. Default: None.
        test_mode (bool): If test_mode=True, gt wouldn't be loaded.
        depth_scale=256: Default KITTI pre-process. divide 256 to get gt measured in meters (m)
        min_depth=1e-3: Default min depth value.
        max_depth=80: Default max depth value.
    """


    def __init__(self,
                 pipeline,
                 img_dir,
                 ann_dir=None,
                 split=None,
                 data_root=None,
                 test_mode=False,
                 depth_scale=256,
                 min_depth=1e-3,
                 max_depth=80):

        self.pipeline = Compose(pipeline)
        self.img_dir = img_dir
        self.ann_dir = ann_dir
        self.split = split
        self.data_root = data_root
        self.test_mode = test_mode
        self.depth_scale = depth_scale
        self.min_depth = min_depth # just for evaluate. (crop gt to certain range)
        self.max_depth = max_depth # just for evaluate.

        # join paths if data_root is specified
        if self.data_root is not None:
            if not (self.img_dir is None or osp.isabs(self.img_dir)):
                self.img_dir = osp.join(self.data_root, self.img_dir)
            if not (self.ann_dir is None or osp.isabs(self.ann_dir)):
                self.ann_dir = osp.join(self.data_root, self.ann_dir)
            if not (self.split is None or osp.isabs(self.split)):
                self.split = osp.join(self.data_root, self.split)

        # load annotations
        self.img_infos = self.load_annotations(self.img_dir, self.ann_dir, self.split)
        
        # load filenames of images to be visualised
        with open('/cluster/project/infk/courses/252-0579-00L/group26/sniall/Monocular-Depth-Estimation-Toolbox/waymo_angled_vis.txt') as f:
            self.vis_files = [line.rstrip() for line in f]
            # print(self.vis_files)

    def __len__(self):
        """Total number of samples of data."""
        return len(self.img_infos)

    def load_annotations(self, img_dir, ann_dir, split):
        """Load annotation from directory.
        Args:
            img_dir (str): Path to image directory
            ann_dir (str|None): Path to annotation directory.
            split (str|None): Split txt file. Split should be specified, only file in the splits will be loaded.
        Returns:
            list[dict]: All image info of dataset.
        """

        self.invalid_depth_num = 0
        img_infos = []
        if split is not None:
            with open(split) as f:
                for line in f:
                    img_info = dict()
                    if ann_dir is not None: # benchmark test or unsupervised future
                        depth_map = line.strip().split(" ")[1]
                        if depth_map == 'None':
                            self.invalid_depth_num += 1
                            continue
                        img_info['ann'] = dict(depth_map=depth_map)
                    img_name = line.strip().split(" ")[0]
                    img_info['filename'] = img_name
                    img_infos.append(img_info)
        else:
            print("Split should be specified, NotImplementedError")
            raise NotImplementedError 

        # github issue:: make sure the same order
        img_infos = sorted(img_infos, key=lambda x: x['filename'])
        print_log(f'Loaded {len(img_infos)} images. Totally {self.invalid_depth_num} invalid pairs are filtered', logger=get_root_logger())

        return img_infos

    def get_ann_info(self, idx):
        """Get annotation by index.
        Args:
            idx (int): Index of data.
        Returns:
            dict: Annotation info of specified index.
        """

        return self.img_infos[idx]['ann']

    def pre_pipeline(self, results):
        """Prepare results dict for pipeline."""
        results['depth_fields'] = []
        results['img_prefix'] = self.img_dir
        results['depth_prefix'] = self.ann_dir
        results['depth_scale'] = self.depth_scale

    def __getitem__(self, idx):
        """Get training/test data after pipeline.
        Args:
            idx (int): Index of data.
        Returns:
            dict: Training/test data (with annotation if `test_mode` is set
                False).
        """

        if self.test_mode:
            return self.prepare_test_img(idx)
        else:
            return self.prepare_train_img(idx)

    def prepare_train_img(self, idx):
        """Get training data and annotations after pipeline.
        Args:
            idx (int): Index of data.
        Returns:
            dict: Training data and annotation after pipeline with new keys
                introduced by pipeline.
        """

        img_info = self.img_infos[idx]
        ann_info = self.get_ann_info(idx)
        results = dict(img_info=img_info, ann_info=ann_info)
        self.pre_pipeline(results)
        return self.pipeline(results)

    def prepare_test_img(self, idx):
        """Get testing data after pipeline.
        Args:
            idx (int): Index of data.
        Returns:
            dict: Testing data after pipeline with new keys introduced by
                pipeline.
        """

        img_info = self.img_infos[idx]
        results = dict(img_info=img_info)
        self.pre_pipeline(results)
        return self.pipeline(results)

    def format_results(self, results, imgfile_prefix=None, indices=None, **kwargs):
        """Place holder to format result to dataset specific output."""
        results[0] = (results[0] * self.depth_scale).astype(np.uint16)
        return results

    def get_gt_depth_maps(self):
        """Get ground truth depth maps for evaluation."""
        for img_info in self.img_infos:
            depth_map = osp.join(self.ann_dir, img_info['ann']['depth_map'])
            depth_map_gt = np.asarray(Image.open(depth_map), dtype=np.float32) / self.depth_scale
            yield depth_map_gt
    
    def get_gt_depth_map(self, path):
        depth_map_gt = np.asarray(Image.open(path), dtype=np.float32) / self.depth_scale
        depth_map_gt = self.downsample_sparse(depth_map_gt, (376, 564))
        return depth_map_gt
    
    def get_unique_identifier(self, path):
        return path.split("/")[-3] + "_" + path.split("/")[-1]
    
    def is_visualize(self, identifier):
        return (identifier in self.vis_files)
    
    def eval_mask(self, depth_gt):
        # depth_gt = np.squeeze(depth_gt)
        valid_mask = np.logical_and(depth_gt > self.min_depth, depth_gt < self.max_depth)
        return valid_mask

    def eval_mask(self, depth_gt):
        """Following Adabins, Do grag_crop or eigen_crop for testing"""
        depth_gt = np.squeeze(depth_gt)
        valid_mask = np.logical_and(depth_gt > self.min_depth, depth_gt < self.max_depth)
        valid_mask = np.expand_dims(valid_mask, axis=0)
        return valid_mask

    def maxpool2d(self, A, kernel_size, stride):
        '''
        2D Pooling

        Parameters:
            A: input 2D array
            kernel_size: int, the size of the window over which we take pool
            stride: int, the stride of the window
        '''

        # Window view of A
        output_shape = ((A.shape[0] - kernel_size) // stride + 1,
                        (A.shape[1] - kernel_size) // stride + 1)
        
        shape_w = (output_shape[0], output_shape[1], kernel_size, kernel_size)
        strides_w = (stride*A.strides[0], stride*A.strides[1], A.strides[0], A.strides[1])
        
        A_w = np.lib.stride_tricks.as_strided(A, shape_w, strides_w)

        # Return the result of pooling
        return A_w.max(axis=(2, 3))
    
    def downsample_sparse(self, depth, size):
        indices = np.nonzero(depth[...])

        new_indices = np.floor(indices[0] * size[0] / depth.shape[0]).astype(np.uint32), \
                      np.floor(indices[1] * size[1] / depth.shape[1]).astype(np.uint32)

        new_depth = np.zeros(size)
        new_depth[new_indices] = depth[indices]

        return new_depth

    def pre_eval(self, preds, indices):
        """Collect eval result from each iteration.
        Args:
            preds (list[torch.Tensor] | torch.Tensor): the depth estimation.
            indices (list[int] | int): the prediction related ground truth
                indices.
        Returns:
            list[torch.Tensor]: (area_intersect, area_union, area_prediction,
                area_ground_truth).
        """
        # In order to compat with batch inference
        if not isinstance(indices, list):
            indices = [indices]
        if not isinstance(preds, list):
            preds = [preds]

        pre_eval_results = []
        pre_eval_preds = []
        pre_eval_gts = []
        pre_eval_masks = []

        for i, (pred, index) in enumerate(zip(preds, indices)):
            depth_map = osp.join(self.ann_dir,
                               self.img_infos[index]['ann']['depth_map'])

            depth_map_gt = cv2.imread(depth_map)
            depth_map_gt = self.downsample_sparse(depth_map_gt, (376, 564))
            # depth_map_gt = self.downsample_sparse(depth_map_gt, (1280 // 2, 1920 // 2))
            # depth_map_gt = self.maxpool2d(depth_map_gt, 3, 3)
            depth_map_gt = depth_map_gt[None, ...]
            valid_mask = np.logical_and(depth_map_gt > self.min_depth, depth_map_gt < self.max_depth)

            pre_eval_gts.append(depth_map_gt)
            pre_eval_masks.append(valid_mask)
            
            eval = metrics(depth_map_gt[valid_mask], 
                           pred[valid_mask], 
                           min_depth=self.min_depth,
                           max_depth=self.max_depth)

            pre_eval_results.append(eval)

            # save prediction results
            pre_eval_preds.append(pred)

        return pre_eval_results, pre_eval_preds, pre_eval_gts, pre_eval_masks

    def evaluate(self, results, metric='eigen', logger=None, **kwargs):
        """Evaluate the dataset.
        Args:
            results (list[tuple[torch.Tensor]] | list[str]): per image pre_eval
                 results or predict depth map for computing evaluation
                 metric.
            logger (logging.Logger | None | str): Logger used for printing
                related information during evaluation. Default: None.
        Returns:
            dict[str, float]: Default metrics.
        """
        metric = ["a1", "a2", "a3", "abs_rel", "rmse", "log_10", "rmse_log", "silog", "sq_rel"]
        
        eval_results = {}
        # test a list of files
        if mmcv.is_list_of(results, np.ndarray) or mmcv.is_list_of(
                results, str):
            gt_depth_maps = self.get_gt_depth_maps()
            ret_metrics = eval_metrics(
                gt_depth_maps,
                results)
        # test a list of pre_eval_results
        else:
            ret_metrics = pre_eval_to_metrics(results)
        
        ret_metric_names = []
        ret_metric_values = []
        for ret_metric, ret_metric_value in ret_metrics.items():
            ret_metric_names.append(ret_metric)
            ret_metric_values.append(ret_metric_value)

        num_table = len(ret_metrics) // 9
        for i in range(num_table):
            names = ret_metric_names[i*9: i*9 + 9]
            values = ret_metric_values[i*9: i*9 + 9]

            # summary table
            ret_metrics_summary = OrderedDict({
                ret_metric: np.round(np.nanmean(ret_metric_value), 4)
                for ret_metric, ret_metric_value in zip(names, values)
            })

            # for logger
            summary_table_data = PrettyTable()
            for key, val in ret_metrics_summary.items():
                summary_table_data.add_column(key, [val])

            print_log('Summary:', logger)
            print_log('\n' + summary_table_data.get_string(), logger=logger)

        # each metric dict
        for key, value in ret_metrics.items():
            eval_results[key] = value

        return eval_results
