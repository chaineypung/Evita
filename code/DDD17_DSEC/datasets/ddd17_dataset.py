# # ---------------------------------------------------------------
# # Code adapted from https://github.com/uzh-rpg/ess/blob/main/datasets/ddd17_events_loader.py.
# # ---------------------------------------------------------------

# import glob
# from os.path import join, exists, dirname, basename
# import os
# import cv2
# import torch
# import numpy as np
# from torch.utils.data import Dataset
# import torchvision.transforms as transforms

# from datasets.extract_data_tools.example_loader_ddd17 import load_files_in_directory, extract_events_from_memmap
# import datasets.data_util as data_util
# import albumentations as A
# from PIL import Image


# def get_split(dirs, split):
#     return {
#         "train": [dirs[0], dirs[2], dirs[3], dirs[5], dirs[6]],
#         "test": [dirs[1]]
#     }[split]


# def unzip_segmentation_masks(dirs):
#     for d in dirs:
#         assert exists(join(d, "segmentation_masks.zip"))
#         if not exists(join(d, "segmentation_masks")):
#             print("Unzipping segmentation mask in %s" % d)
#             os.system("unzip %s -d %s" % (join(d, "segmentation_masks"), d))


# class DDD17Event(Dataset):
#     def __init__(self, root, split="train", event_representation='voxel_grid',
#                  nr_events_data=1, delta_t_per_data=50, nr_bins_per_data=5, require_paired_data=False,
#                  separate_pol=False, normalize_event=False, augmentation=False, fixed_duration=True,
#                  nr_events_per_data=32000, random_crop=False):
#         data_dirs = sorted(glob.glob(join(root, "dir*")))
#         assert len(data_dirs) > 0
#         assert split in ["train", "test"]

#         self.split = split  # 数据集划分（训练或测试）
#         self.augmentation = augmentation  # 是否进行数据增强
#         self.fixed_duration = fixed_duration  # 是否使用固定时间间隔
#         self.nr_events_per_data = nr_events_per_data  # 每段数据的事件数量

#         self.nr_events_data = nr_events_data  # 每段数据的事件块数量
#         self.delta_t_per_data = delta_t_per_data  # 每段数据的时间间隔
#         if self.fixed_duration:
#             self.t_interval = nr_events_data * delta_t_per_data  # 固定时间间隔
#         else:
#             self.t_interval = -1  # 不固定时间间隔
#             self.nr_events = self.nr_events_data * self.nr_events_per_data  # 总事件数量
#         assert self.t_interval in [10, 50, 250, -1]  # 确保时间间隔合法
#         self.nr_temporal_bins = nr_bins_per_data  # 时间维度的分块数量
#         self.require_paired_data = require_paired_data  # 是否需要事件-图像对
#         self.event_representation = event_representation  # 事件表示方式
#         self.shape = [260, 346]  # 数据形状
#         self.random_crop = random_crop  # 是否随机裁剪
#         self.shape_crop = [200, 346]  # 裁剪后的形状
#         self.separate_pol = separate_pol  # 是否分离极性
#         self.normalize_event = normalize_event  # 是否归一化事件数据
#         self.dirs = get_split(data_dirs, split)  # 根据split获取对应的数据目录
#         # unzip_segmentation_masks(self.dirs)

#         self.files = []
#         for d in self.dirs:
#             self.files += sorted(glob.glob(join(d, "segmentation_masks", "*.png")))
#         print("[DDD17Event]: Found %s segmentation masks for split %s" % (len(self.files), split))

#         # load events and image_idx -> event index mapping
#         self.img_timestamp_event_idx = {}
#         self.event_data = {}

#         print("[DDD17Event]: Loading real events.")
#         self.event_dirs = self.dirs

#         for d in self.event_dirs:
#             img_timestamp_event_idx, t_events, xyp_events, _ = load_files_in_directory(d, self.t_interval)
#             self.img_timestamp_event_idx[d] = img_timestamp_event_idx
#             self.event_data[d] = [t_events, xyp_events]

#         if self.augmentation:
#             self.transform_a = A.ReplayCompose([
#                 A.HorizontalFlip(p=0.5)
#             ])
#             self.transform_a_random_crop = A.ReplayCompose([
#                 A.RandomScale(scale_limit=(0, 0.8), p=1),
#                 A.RandomCrop(height=self.shape_crop[0], width=self.shape_crop[1], always_apply=True),
#                 A.HorizontalFlip(p=0.5)])
#         self.transform_a_center_crop = A.ReplayCompose([
#             A.CenterCrop(height=self.shape_crop[0], width=self.shape_crop[1], always_apply=True),
#         ])

#     def __len__(self):
#         return len(self.files)

#     def apply_augmentation(self, transform_a, events, images, label):
#         if self.require_paired_data:    # 对Event-Image pair进行数据增强
#             A_data = transform_a(image=images.permute(1, 2, 0).numpy(), mask=label)
#             img_tensor = torch.from_numpy(A_data['image']).permute(2, 0, 1)
#             label = A_data['mask']
#             if self.random_crop and self.split == 'train':
#                 events_tensor = torch.zeros((events.shape[0], self.shape_crop[0], self.shape_crop[1]))
#             else:
#                 events_tensor = events
#             for k in range(events.shape[0]):
#                 events_tensor[k, :, :] = torch.from_numpy(
#                     A.ReplayCompose.replay(A_data['replay'], image=events[k, :, :].numpy())['image'])
#             return events_tensor, img_tensor, label
#         else:
#             A_data = transform_a(image=events[0, :, :].numpy(), mask=label)
#             label = A_data['mask']
#             if self.random_crop and self.split == 'train':
#                 events_tensor = torch.zeros((events.shape[0], self.shape_crop[0], self.shape_crop[1]))
#             else:
#                 events_tensor = events
#             for k in range(events.shape[0]):
#                 events_tensor[k, :, :] = torch.from_numpy(
#                     A.ReplayCompose.replay(A_data['replay'], image=events[k, :, :].numpy())['image'])
#             return events_tensor, label

#     def __getitem__(self, idx):
#         segmentation_mask_file = self.files[idx]
#         segmentation_mask = cv2.imread(segmentation_mask_file, 0)
#         label = np.array(segmentation_mask)
#         directory = dirname(dirname(segmentation_mask_file))
#         img_idx = int(basename(segmentation_mask_file).split("_")[-1].split(".")[0]) - 1
#         img_timestamp_event_idx = self.img_timestamp_event_idx[directory]
#         t_events, xyp_events = self.event_data[directory]

#         # events has form x, y, t_ns, p (in [0,1])
#         if self.fixed_duration:
#             events = extract_events_from_memmap(t_events, xyp_events, img_idx, img_timestamp_event_idx, self.fixed_duration)
#         else:
#             events = extract_events_from_memmap(t_events, xyp_events, img_idx, img_timestamp_event_idx,
#                                                 self.fixed_duration, self.nr_events)
#         t_ns = events[:, 2]
#         delta_t_ns = int((t_ns[-1] - t_ns[0]) / self.nr_events_data)
#         nr_events_loaded = events.shape[0]
#         nr_events_temp = nr_events_loaded // self.nr_events_data

#         id_end = 0
        
#         # Generate the event-image pair
#         img_tensor = None
#         if self.require_paired_data:
#             segmentation_mask_filepath_list = str(segmentation_mask_file).split('/')
#             segmentation_mask_filename = segmentation_mask_filepath_list[-1]
#             filename_id = segmentation_mask_filename.split('_')[-1]
#             img_filename = '_'.join(['img', filename_id])
#             img_filepath_list = segmentation_mask_filepath_list
#             img_filepath_list[-2] = 'imgs'
#             img_filepath_list[-1] = img_filename
#             img_file = '/'.join(img_filepath_list)
#             if not os.path.exists(img_file):
#                 img_filename = filename_id.zfill(14)
#                 img_filepath_list[-1] = img_filename
#                 img_file = '/'.join(img_filepath_list)
#             img = Image.open(img_file)

#             img_transform = transforms.Compose([
#                 transforms.Grayscale(),
#                 transforms.ToTensor()
#             ])
#             img_tensor = img_transform(img)
#             # img_tensor = img_tensor[:, :-60, :]

#         # Generate the event tensor
#         event_tensor = None
#         for i in range(self.nr_events_data):
#             id_start = id_end
#             if self.fixed_duration:
#                 id_end = np.searchsorted(t_ns, t_ns[0] + (i + 1) * delta_t_ns)
#             else:
#                 id_end += nr_events_temp

#             if id_end > nr_events_loaded:
#                 id_end = nr_events_loaded

#             event_representation = data_util.generate_input_representation(events[id_start:id_end],
#                                                                            self.event_representation,
#                                                                            self.shape,
#                                                                            nr_temporal_bins=self.nr_temporal_bins,
#                                                                            separate_pol=self.separate_pol, 
#                                                                            img=img_tensor)

#             event_representation = torch.from_numpy(event_representation)

#             if self.normalize_event:
#                 event_representation = data_util.normalize_voxel_grid(event_representation)

#             if event_tensor is None:
#                 event_tensor = event_representation
#             else:
#                 event_tensor = torch.cat([event_tensor, event_representation], dim=0)

#         event_tensor = event_tensor[:, :-60, :]  # remove 60 bottom rows
#         img_tensor = img_tensor[:, :-60, :]     # 处理完event_tensor后，图像tensor也需要去掉底部60行

#         # Data augmentation
#         if self.random_crop and self.split == 'train':
#             if self.augmentation:
#                 if self.require_paired_data:
#                     event_tensor, img_tensor, label = self.apply_augmentation(self.transform_a_random_crop,
#                                                                               event_tensor, img_tensor, label)
#                 else:
#                     event_tensor, label = self.apply_augmentation(self.transform_a_random_crop, event_tensor,
#                                                                   img_tensor, label)
#         else:
#             if self.augmentation:
#                 if self.require_paired_data:
#                     event_tensor, img_tensor, label = self.apply_augmentation(self.transform_a, event_tensor,
#                                                                               img_tensor, label)
#                 else:
#                     event_tensor, label = self.apply_augmentation(self.transform_a, event_tensor, img_tensor, label)

#         label_tensor = torch.from_numpy(label).long()

#         # Return event-image pair data or only event data
#         if self.require_paired_data:
#             return event_tensor, img_tensor, label_tensor
#             # return img_tensor, img_tensor, label_tensor
#         else:
#             return event_tensor, label_tensor



# ---------------------------------------------------------------
# Code adapted from https://github.com/uzh-rpg/ess/blob/main/datasets/ddd17_events_loader.py.
# ---------------------------------------------------------------

import glob
from os.path import join, exists, dirname, basename
import os
import cv2
import torch
import numpy as np
import random
from torch.utils.data import Dataset
import torchvision.transforms as transforms

from datasets.extract_data_tools.example_loader_ddd17 import load_files_in_directory, extract_events_from_memmap
import datasets.data_util as data_util
import albumentations as A
from PIL import Image


def get_split(dirs, split):
    return {
        "train": [dirs[0], dirs[2], dirs[3], dirs[5], dirs[6]],
        "test": [dirs[1]]
    }[split]


def unzip_segmentation_masks(dirs):
    for d in dirs:
        assert exists(join(d, "segmentation_masks.zip"))
        if not exists(join(d, "segmentation_masks")):
            print("Unzipping segmentation mask in %s" % d)
            os.system("unzip %s -d %s" % (join(d, "segmentation_masks"), d))


class DDD17Event(Dataset):
    def __init__(self, root, split="train", event_representation='voxel_grid',
                 nr_events_data=1, delta_t_per_data=50, nr_bins_per_data=5, require_paired_data=False,
                 separate_pol=False, normalize_event=False, augmentation=False, fixed_duration=True,
                 nr_events_per_data=32000, random_crop=False, 
                 mosaic_ratio=0.00): # Added mosaic_ratio parameter
        
        data_dirs = sorted(glob.glob(join(root, "dir*")))
        assert len(data_dirs) > 0
        assert split in ["train", "test"]

        self.split = split  # Dataset split (train or test)
        self.augmentation = augmentation  # Whether to apply data augmentation
        self.fixed_duration = fixed_duration  # Whether to use fixed time duration
        self.nr_events_per_data = nr_events_per_data  # Number of events per data chunk

        # Mosaic probability
        self.mosaic_ratio = mosaic_ratio

        self.nr_events_data = nr_events_data  # Number of event blocks per data
        self.delta_t_per_data = delta_t_per_data  # Time interval per data
        
        if self.fixed_duration:
            self.t_interval = nr_events_data * delta_t_per_data  # Fixed time interval
        else:
            self.t_interval = -1  # Variable time interval
            self.nr_events = self.nr_events_data * self.nr_events_per_data  # Total number of events
        
        assert self.t_interval in [10, 50, 250, -1]  # Ensure valid time interval
        
        self.nr_temporal_bins = nr_bins_per_data  # Number of temporal bins
        self.require_paired_data = require_paired_data  # Whether paired event-image data is required
        self.event_representation = event_representation  # Event representation type
        self.shape = [260, 346]  # Original data shape
        self.random_crop = random_crop  # Whether to apply random cropping
        self.shape_crop = [200, 346]  # Shape after cropping
        self.separate_pol = separate_pol  # Whether to separate polarity
        self.normalize_event = normalize_event  # Whether to normalize event data
        self.dirs = get_split(data_dirs, split)  # Get data directories based on split

        self.files = []
        for d in self.dirs:
            self.files += sorted(glob.glob(join(d, "segmentation_masks", "*.png")))
        print("[DDD17Event]: Found %s segmentation masks for split %s" % (len(self.files), split))

        # Load events and image_idx -> event index mapping
        self.img_timestamp_event_idx = {}
        self.event_data = {}

        print("[DDD17Event]: Loading real events.")
        self.event_dirs = self.dirs

        for d in self.event_dirs:
            img_timestamp_event_idx, t_events, xyp_events, _ = load_files_in_directory(d, self.t_interval)
            self.img_timestamp_event_idx[d] = img_timestamp_event_idx
            self.event_data[d] = [t_events, xyp_events]

        if self.augmentation:
            self.transform_a = A.ReplayCompose([
                A.HorizontalFlip(p=0.5)
            ])
            # If using Mosaic, we usually do not need RandomScale/RandomCrop again here
            self.transform_a_random_crop = A.ReplayCompose([
                A.RandomScale(scale_limit=(0, 0.8), p=1),
                A.RandomCrop(height=self.shape_crop[0], width=self.shape_crop[1], always_apply=True),
                A.HorizontalFlip(p=0.5)])
        
        self.transform_a_center_crop = A.ReplayCompose([
            A.CenterCrop(height=self.shape_crop[0], width=self.shape_crop[1], always_apply=True),
        ])

    def __len__(self):
        return len(self.files)

    def _load_raw_sample(self, idx):
        """
        Helper function: Responsible for loading raw data (removing bottom 60 rows),
        without applying augmentation. Returns numpy arrays.
        """
        segmentation_mask_file = self.files[idx]
        segmentation_mask = cv2.imread(segmentation_mask_file, 0)
        label = np.array(segmentation_mask)
        directory = dirname(dirname(segmentation_mask_file))
        img_idx = int(basename(segmentation_mask_file).split("_")[-1].split(".")[0]) - 1
        img_timestamp_event_idx = self.img_timestamp_event_idx[directory]
        t_events, xyp_events = self.event_data[directory]

        # Extract events (x, y, t_ns, p)
        if self.fixed_duration:
            events = extract_events_from_memmap(t_events, xyp_events, img_idx, img_timestamp_event_idx, self.fixed_duration)
        else:
            events = extract_events_from_memmap(t_events, xyp_events, img_idx, img_timestamp_event_idx,
                                                self.fixed_duration, self.nr_events)
        t_ns = events[:, 2]
        delta_t_ns = int((t_ns[-1] - t_ns[0]) / self.nr_events_data)
        nr_events_loaded = events.shape[0]
        nr_events_temp = nr_events_loaded // self.nr_events_data

        id_end = 0
        
        # Load Image
        img_tensor = None
        if self.require_paired_data:
            segmentation_mask_filepath_list = str(segmentation_mask_file).split('/')
            segmentation_mask_filename = segmentation_mask_filepath_list[-1]
            filename_id = segmentation_mask_filename.split('_')[-1]
            img_filename = '_'.join(['img', filename_id])
            img_filepath_list = segmentation_mask_filepath_list
            img_filepath_list[-2] = 'imgs'
            img_filepath_list[-1] = img_filename
            img_file = '/'.join(img_filepath_list)
            if not os.path.exists(img_file):
                img_filename = filename_id.zfill(14)
                img_filepath_list[-1] = img_filename
                img_file = '/'.join(img_filepath_list)
            img = Image.open(img_file)

            img_transform = transforms.Compose([
                transforms.Grayscale(),
                transforms.ToTensor()
            ])
            img_tensor = img_transform(img) # [1, H, W]

        # Generate Event Tensor
        event_tensor = None
        for i in range(self.nr_events_data):
            id_start = id_end
            if self.fixed_duration:
                id_end = np.searchsorted(t_ns, t_ns[0] + (i + 1) * delta_t_ns)
            else:
                id_end += nr_events_temp

            if id_end > nr_events_loaded:
                id_end = nr_events_loaded

            event_representation = data_util.generate_input_representation(events[id_start:id_end],
                                                                           self.event_representation,
                                                                           self.shape,
                                                                           nr_temporal_bins=self.nr_temporal_bins,
                                                                           separate_pol=self.separate_pol,
                                                                           img=img_tensor)
            event_representation = torch.from_numpy(event_representation)

            if self.normalize_event:
                event_representation = data_util.normalize_voxel_grid(event_representation)

            if event_tensor is None:
                event_tensor = event_representation
            else:
                event_tensor = torch.cat([event_tensor, event_representation], dim=0)

        # Remove bottom 60 rows [C, H, W]
        event_tensor = event_tensor[:, :-60, :] 
        if img_tensor is not None:
            img_tensor = img_tensor[:, :-60, :]
        
        # Return as numpy for easier slicing in Mosaic
        # event: (C, H, W), img: (1, H, W), label: (H, W)
        return event_tensor.numpy(), (img_tensor.numpy() if img_tensor is not None else None), label

    def _load_mosaic(self, index):
        """
        Implementation of Mosaic Data Augmentation.
        Randomly selects 3 additional images to compose a 2x2 grid.
        """
        # Randomly select 3 additional indices
        indexes = [index] + [random.randint(0, len(self.files) - 1) for _ in range(3)]
        
        # Load 4 samples
        data_a = self._load_raw_sample(indexes[0])
        data_b = self._load_raw_sample(indexes[1])
        data_c = self._load_raw_sample(indexes[2])
        data_d = self._load_raw_sample(indexes[3])

        # Unpack data
        evt_a, img_a, mask_a = data_a
        evt_b, img_b, mask_b = data_b
        evt_c, img_c, mask_c = data_c
        evt_d, img_d, mask_d = data_d

        # Target dimensions (H, W) -> self.shape_crop [200, 346]
        target_h, target_w = self.shape_crop
        
        # Original dimensions (H_orig, W_orig) -> [200, 346] (after removing 60 rows)
        orig_h, orig_w = evt_a.shape[1], evt_a.shape[2]
        
        # Determine split center point
        start_x = target_w // 4
        start_y = target_h // 4
        xc = random.randint(start_x, target_w - start_x)
        yc = random.randint(start_y, target_h - start_y)

        # Define positions for 4 quadrants: (x_start, y_start, x_end, y_end)
        pos_a = (0, 0, xc, yc)                  # Top-Left
        pos_b = (xc, 0, target_w, yc)           # Top-Right
        pos_c = (0, yc, xc, target_h)           # Bottom-Left
        pos_d = (xc, yc, target_w, target_h)    # Bottom-Right

        # Initialize canvas
        c_evt = evt_a.shape[0]
        final_evt = np.full((c_evt, target_h, target_w), 0, dtype=evt_a.dtype)
        final_mask = np.full((target_h, target_w), 0, dtype=mask_a.dtype)
        
        final_img = None
        if self.require_paired_data:
            c_img = img_a.shape[0]
            final_img = np.full((c_img, target_h, target_w), 0, dtype=img_a.dtype)

        # Helper function to place a random patch into the canvas
        def place_patch(src_evt, src_img, src_mask, pos):
            x1, y1, x2, y2 = pos
            w_needed = x2 - x1
            h_needed = y2 - y1
            
            # Safe crop dimensions
            w_crop = min(w_needed, orig_w)
            h_crop = min(h_needed, orig_h)
            
            # Random crop from source
            sx = random.randint(0, orig_w - w_crop)
            sy = random.randint(0, orig_h - h_crop)
            
            # Assign to canvas
            final_evt[:, y1:y1+h_crop, x1:x1+w_crop] = src_evt[:, sy:sy+h_crop, sx:sx+w_crop]
            final_mask[y1:y1+h_crop, x1:x1+w_crop] = src_mask[sy:sy+h_crop, sx:sx+w_crop]
            
            if src_img is not None and final_img is not None:
                final_img[:, y1:y1+h_crop, x1:x1+w_crop] = src_img[:, sy:sy+h_crop, sx:sx+w_crop]

        # Execute placement
        place_patch(evt_a, img_a, mask_a, pos_a)
        place_patch(evt_b, img_b, mask_b, pos_b)
        place_patch(evt_c, img_c, mask_c, pos_c)
        place_patch(evt_d, img_d, mask_d, pos_d)

        # Convert back to Tensor
        event_tensor = torch.from_numpy(final_evt)
        label_tensor = torch.from_numpy(final_mask).long()
        img_tensor = torch.from_numpy(final_img) if final_img is not None else None
        
        return event_tensor, img_tensor, label_tensor

    def apply_augmentation(self, transform_a, events, images, label):
        """
        Apply standard augmentations (Flip, Crop, etc.) using Albumentations.
        """
        if self.require_paired_data:  # Apply augmentation to Event-Image pair
            A_data = transform_a(image=images.permute(1, 2, 0).numpy(), mask=label)
            img_tensor = torch.from_numpy(A_data['image']).permute(2, 0, 1)
            label = A_data['mask']
            
            if self.random_crop and self.split == 'train':
                events_tensor = torch.zeros((events.shape[0], self.shape_crop[0], self.shape_crop[1]))
            else:
                events_tensor = events
            
            # Replay augmentation on events (channel-wise)
            for k in range(events.shape[0]):
                events_tensor[k, :, :] = torch.from_numpy(
                    A.ReplayCompose.replay(A_data['replay'], image=events[k, :, :].numpy())['image'])
            return events_tensor, img_tensor, label
        else:
            # Apply augmentation only to Event
            A_data = transform_a(image=events[0, :, :].numpy(), mask=label)
            label = A_data['mask']
            
            if self.random_crop and self.split == 'train':
                events_tensor = torch.zeros((events.shape[0], self.shape_crop[0], self.shape_crop[1]))
            else:
                events_tensor = events
            
            for k in range(events.shape[0]):
                events_tensor[k, :, :] = torch.from_numpy(
                    A.ReplayCompose.replay(A_data['replay'], image=events[k, :, :].numpy())['image'])
            return events_tensor, label

    def __getitem__(self, idx):
        # Determine whether to use Mosaic
        use_mosaic = False
        if self.split == 'train' and self.mosaic_ratio > 0.0:
            if random.random() < self.mosaic_ratio:
                use_mosaic = True

        if use_mosaic:
            # 1. Load Mosaic data (Size is already 200x346)
            event_tensor, img_tensor, label_tensor = self._load_mosaic(idx)
            
            # 2. Apply additional augmentation (e.g., Flip)
            # Since Mosaic already acts as a random crop, we use the simple transform (Flip only)
            if self.augmentation:
                if self.require_paired_data:
                    event_tensor, img_tensor, label_tensor = self.apply_augmentation(
                        self.transform_a, event_tensor, img_tensor, label_tensor.numpy())
                else:
                    event_tensor, label_tensor = self.apply_augmentation(
                        self.transform_a, event_tensor, img_tensor, label_tensor.numpy())
            
            # Ensure label is tensor
            if not isinstance(label_tensor, torch.Tensor):
                label_tensor = torch.from_numpy(label_tensor).long()

        else:
            # [Standard Pipeline] Load single sample
            evt_np, img_np, label_np = self._load_raw_sample(idx)
            
            event_tensor = torch.from_numpy(evt_np)
            img_tensor = torch.from_numpy(img_np) if img_np is not None else None
            
            # Select transformation
            if self.random_crop and self.split == 'train':
                if self.augmentation:
                    trans = self.transform_a_random_crop
                else:
                    trans = self.transform_a_center_crop
            else:
                 trans = self.transform_a_center_crop

            # Apply augmentation
            if self.require_paired_data:
                event_tensor, img_tensor, label_np = self.apply_augmentation(
                    trans, event_tensor, img_tensor, label_np)
            else:
                event_tensor, label_np = self.apply_augmentation(
                    trans, event_tensor, img_tensor, label_np)
            
            label_tensor = torch.from_numpy(label_np).long()

        # Return event-image pair data or only event data
        if self.require_paired_data:
            return event_tensor, img_tensor, label_tensor
        else:
            return event_tensor, label_tensor