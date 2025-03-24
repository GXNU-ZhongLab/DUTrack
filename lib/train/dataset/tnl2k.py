import os
import os.path
import torch
import numpy as np
import pandas
import csv
import random
from collections import OrderedDict
from .base_video_dataset import BaseVideoDataset
from lib.train.data import jpeg4py_loader_w_failsafe
import re
import json
import bisect
# from lib.utils.string_utils import clean_string


class TNL2k(BaseVideoDataset):
    """ TNL2k dataset.

    Publication: Towards More Flexible and Accurate Object Tracking with Natural Language: Algorithms and Benchmark (CVPR 2021)
    authors:     Xiao Wang, Xiujun Shu, Zhipeng Zhang, Bo Jiang, Yaowei Wang, Yonghong Tian, Feng Wu
    Download the dataset from https://github.com/wangxiao5791509/TNL2K_evaluation_toolkit
    """

    def __init__(self, root=None, image_loader=jpeg4py_loader_w_failsafe, split=None, data_fraction=None):
        """
        args:
            root - path to the lasot dataset.
            image_loader (jpeg4py_loader) -  The function to read the images. jpeg4py (https://github.com/ajkxyz/jpeg4py)
                                            is used by default.
            vid_ids - List containing the ids of the videos (1 - 20) used for training. If vid_ids = [1, 3, 5], then the
                    videos with subscripts -1, -3, and -5 from each class will be used for training.
            split - If split='train', the official train split (protocol-II) is used for training. Note: Only one of
                    vid_ids or split option can be used at a time.
            data_fraction - Fraction of dataset to be used. The complete dataset is used by default
        """
        self.root = root + '/' + split
        super().__init__('tnl2k', self.root, image_loader)

        # Keep a list of all classes
        self.sequence_list = self._build_sequence_list()

        self.data_index = self._create_data_index()

        if data_fraction is not None:
            self.sequence_list = random.sample(self.sequence_list, int(len(self.sequence_list) * data_fraction))

    def _build_sequence_list(self):
        sequence_list = []
        subset_list = [f for f in os.listdir(self.root)
                       if os.path.isdir(os.path.join(self.root, f)) and f != 'revised_annotations']

        # one-level directory: ['INF_womanleft', 'INF_whitesuv', ...]
        if len(subset_list) > 14:
            self.dir_type = 'one-level'
            return sorted(subset_list)

        # two-level directory: ['TNL2k_train_subset_p9/INF_womanleft', 'TNL2k_train_subset_p9/INF_whitesuv', ...]
        self.dir_type = 'two-level'
        for x in subset_list:
            sub_sequence_list_path = os.path.join(self.root, x)
            for seq in os.listdir(sub_sequence_list_path):
                sequence_list.append(os.path.join(x, seq))
        sequence_list = sorted(sequence_list)
        return sequence_list

    def _create_data_index(self):
        tnl2k_cache_root = "/home/local_data/lxh/code/LoRAVL/lib/train/data_specs/tnl2k_index"
        if not os.path.exists(tnl2k_cache_root):
            os.makedirs(tnl2k_cache_root)
        if os.path.exists(os.path.join(tnl2k_cache_root, 'index.json')):
            with open(os.path.join(tnl2k_cache_root, 'index.json'), "r") as f:
                tnl2k_index = json.load(f)
        else:
            tnl2k_index = {}
            print("saving index for tnl2k...")
            for seq in self.sequence_list:
                img_list = os.listdir(os.path.join(self.root, seq, 'imgs'))
                img_list = self._sort(img_list)
                tnl2k_index[seq] = img_list
            with open(os.path.join(tnl2k_cache_root, 'index.json'), "w") as f:
                json.dump(tnl2k_index, f)

        return tnl2k_index

    def get_name(self):
        return 'tnl2k'

    def has_class_info(self):
        return True

    def has_occlusion_info(self):
        return True

    def get_num_sequences(self):
        return len(self.sequence_list)

    def get_num_classes(self):
        return len(self.class_list)

    def get_sequences_in_class(self, class_name):
        return self.seq_per_class[class_name]

    def _read_bb_anno(self, seq_path):
        bb_anno_file = os.path.join(seq_path, "groundtruth.txt")
        gt = pandas.read_csv(bb_anno_file, delimiter=',', header=None, dtype=np.float32, na_filter=False,
                             low_memory=False).values
        return torch.tensor(gt)

    def _get_sequence_path(self, seq_id):
        if self.dir_type == 'two-level':
            seq_name = self.sequence_list[seq_id].split('/')[-1]
            class_name = self.sequence_list[seq_id].split('/')[0]
            return os.path.join(self.root, class_name, seq_name)
        else:
            seq_name = self.sequence_list[seq_id]
            return os.path.join(self.root, seq_name)

    def _get_sequence_name(self, seq_id):
        seq_name = self.sequence_list[seq_id]
        return seq_name

    def get_sequence_info(self, seq_id):
        seq_path = self._get_sequence_path(seq_id)
        bbox = self._read_bb_anno(seq_path)

        valid = (bbox[:, 2] > 0) & (bbox[:, 3] > 0)
        visible = valid.byte()

        return {'bbox': bbox, 'valid': valid, 'visible': visible}

    def _sort(self, x, s='.'):
        x.sort(key=lambda x: int(re.sub('[a-zA-Z]', '', x.split(s)[0])))
        return x

    def _get_frame_path(self, seq_path, seq_name, frame_id):
        # should speed up
        # img_list = os.listdir(os.path.join(seq_path, 'imgs'))
        # img_list = self._sort(img_list)
        img_list = self.data_index[seq_name]
        try:
            img_name = img_list[frame_id]
        except Exception as e:
            print('ERROR: Could not find image "{}"'.format(os.path.join(seq_path, 'imgs', img_name)))
            print(e)
            return None

        return os.path.join(seq_path, 'imgs', img_name)

    def _get_frame(self, seq_path, seq_name, frame_id):
        return self.image_loader(self._get_frame_path(seq_path, seq_name, frame_id))

    def _get_class(self, seq_path):
        raw_class = seq_path.split('/')[-2]
        return raw_class

    def get_class_name(self, seq_id):
        seq_path = self._get_sequence_path(seq_id)
        obj_class = self._get_class(seq_path)

        return obj_class

    def get_frames(self, seq_id, frame_ids, anno=None):
        seq_path = self._get_sequence_path(seq_id)
        seq_name = self._get_sequence_name(seq_id)

        obj_class = self._get_class(seq_path)
        frame_list = [self._get_frame(seq_path, seq_name, f_id) for f_id in frame_ids]

        if anno is None:
            anno = self.get_sequence_info(seq_id)

        anno_frames = {}
        for key, value in anno.items():
            anno_frames[key] = [value[f_id, ...].clone() for f_id in frame_ids]

        object_meta = OrderedDict({'object_class_name': obj_class,
                                   'motion_class': None,
                                   'major_class': None,
                                   'root_class': None,
                                   'motion_adverb': None})

        return frame_list, anno_frames, object_meta


class TNL2k_Lang(TNL2k):
    """ TNL2k dataset.

    Publication: Towards More Flexible and Accurate Object Tracking with Natural Language: Algorithms and Benchmark (CVPR 2021)
    authors:     Xiao Wang, Xiujun Shu, Zhipeng Zhang, Bo Jiang, Yaowei Wang, Yonghong Tian, Feng Wu
    Download the dataset from https://github.com/wangxiao5791509/TNL2K_evaluation_toolkit
    """

    def __init__(self, root=None, image_loader=jpeg4py_loader_w_failsafe, split=None, data_fraction=None):
        """
        args:
            root - path to the lasot dataset.
            image_loader (jpeg4py_loader) -  The function to read the images. jpeg4py (https://github.com/ajkxyz/jpeg4py)
                                            is used by default.
            vid_ids - List containing the ids of the videos (1 - 20) used for training. If vid_ids = [1, 3, 5], then the
                    videos with subscripts -1, -3, and -5 from each class will be used for training.
            split - If split='train', the official train split (protocol-II) is used for training. Note: Only one of
                    vid_ids or split option can be used at a time.
            data_fraction - Fraction of dataset to be used. The complete dataset is used by default
        """
        super().__init__(root=root, image_loader=image_loader, split=split, data_fraction=data_fraction)

    def get_language_description(self, seq_id, frame_ids):
        seq_name = self.sequence_list[seq_id]
        root = self.root.split('/train')[0]
        txt_path = os.path.join(root, "tnl2k_train_concise", seq_name + '.txt')
        result_dict = {}
        with open(txt_path, 'r') as file:
            for line in file:
                key, value = line.strip().split(' ', 1)
                result_dict[int(key)] = value
        keys = sorted(result_dict.keys())
        idx = bisect.bisect_left(keys, frame_ids)
        if idx == 0:
            id = keys[0]
        elif idx == len(keys):
            id = keys[-1]
        else:
            id = keys[idx - 1]

        language_description = result_dict[id]
        # txt_path = self._get_txt_path(seq_id)
        return language_description

    def get_frames(self, seq_id, frame_ids, anno=None):
        seq_path = self._get_sequence_path(seq_id)
        seq_name = self._get_sequence_name(seq_id)

        obj_class = self._get_class(seq_path)
        frame_list = [self._get_frame(seq_path, seq_name, f_id) for f_id in frame_ids]

        if anno is None:
            anno = self.get_sequence_info(seq_id)

        anno_frames = {}
        for key, value in anno.items():
            anno_frames[key] = [value[f_id, ...].clone() for f_id in frame_ids]

        language_dict_list =[self.get_language_description(seq_id,f_id) for f_id in frame_ids]

        object_meta = OrderedDict({'object_class_name': obj_class,
                                   'motion_class': None,
                                   'major_class': None,
                                   'root_class': None,
                                   'motion_adverb': None,
                                   'language_description': language_dict_list
                                   })

        return frame_list, anno_frames, object_meta