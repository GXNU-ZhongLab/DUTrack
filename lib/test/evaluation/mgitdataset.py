import glob
import json
import numpy as np
import os
import pandas as pd
import six

import numpy as np
from lib.test.evaluation.data import Sequence, BaseDataset, SequenceList
from lib.test.utils.load_text import load_text


class MGITDataset(BaseDataset):
    """
    LaSOT test set consisting of 280 videos (see Protocol-II in the LaSOT paper)

    Publication:
        LaSOT: A High-quality Benchmark for Large-scale Single Object Tracking
        Heng Fan, Liting Lin, Fan Yang, Peng Chu, Ge Deng, Sijia Yu, Hexin Bai, Yong Xu, Chunyuan Liao and Haibin Ling
        CVPR, 2019
        https://arxiv.org/pdf/1809.07845.pdf

    Download the dataset from https://cis.temple.edu/lasot/download.html
    """
    def __init__(self,version='tiny'):
        super().__init__()
        self.base_path = self.env_settings.mgit_path
        self.version = version
        self.sequence_list = self._get_sequence_list()
        self.clean_list = self.sequence_list

    def clean_seq_list(self):
        clean_lst = []
        for i in range(len(self.sequence_list)):
            cls, _ = self.sequence_list[i].split('-')
            clean_lst.append(cls)
        return  clean_lst

    def get_sequence_list(self):
        return SequenceList([self._construct_sequence(s) for s in self.sequence_list])

    def _construct_sequence(self, sequence_name):
        anno_path = '{}/{}/{}/{}.txt'.format(self.base_path, 'attribute','groundtruth', sequence_name)
        ground_truth_rect = load_text(str(anno_path), delimiter=',', dtype=np.float64, backend='numpy')

        frames_path = '{}/{}/{}/frame_{}'.format(self.base_path, 'MGIT-Test',sequence_name ,sequence_name)
        frame_list = [frame for frame in os.listdir(frames_path) if frame.endswith(".jpg")]
        frame_list.sort(key=lambda f: int(f[:-4]))
        frames_list = [os.path.join(frames_path, frame) for frame in frame_list]

        return Sequence(sequence_name, frames_list, 'mgit', ground_truth_rect.reshape(-1, 4))

        # class_name = sequence_name.split('-')[0]
        # anno_path = '{}/{}/{}/{}.txt'.format(self.base_path, 'attribute','groundtruth', sequence_name)
        # # anno_path = '{}/{}/{}/groundtruth.txt'.format(self.base_path, class_name, sequence_name)
        #
        # ground_truth_rect = load_text(str(anno_path), delimiter=',', dtype=np.float64)
        #
        # occlusion_label_path = '{}/{}/{}/full_occlusion.txt'.format(self.base_path, class_name, sequence_name)
        #
        # # NOTE: pandas backed seems super super slow for loading occlusion/oov masks
        # full_occlusion = load_text(str(occlusion_label_path), delimiter=',', dtype=np.float64, backend='numpy')
        #
        # out_of_view_label_path = '{}/{}/{}/out_of_view.txt'.format(self.base_path, class_name, sequence_name)
        # out_of_view = load_text(str(out_of_view_label_path), delimiter=',', dtype=np.float64, backend='numpy')
        #
        # target_visible = np.logical_and(full_occlusion == 0, out_of_view == 0)
        #
        # frames_path = '{}/{}/{}/img'.format(self.base_path, class_name, sequence_name)
        #
        # frames_list = ['{}/{:08d}.jpg'.format(frames_path, frame_number) for frame_number in range(1, ground_truth_rect.shape[0] + 1)]
        #
        # target_class = class_name
        # return Sequence(sequence_name, frames_list, 'lasot', ground_truth_rect.reshape(-1, 4),
        #                 object_class=target_class, target_visible=target_visible)

    def __len__(self):
        return len(self.sequence_list)

    def _get_sequence_list(self):
        f = open(os.path.join(os.path.split(os.path.realpath(__file__))[0], 'mgit.json'), 'r', encoding='utf-8')
        self.infos = json.load(f)[self.version]
        f.close()
        sequence_list = self.infos['test']
        return sequence_list


class MGIT(object):
    r"""`MGIT <http://videocube.aitestunion.com>`_ Dataset.

    Publication:
        ``A Multi-modal Global Instance Tracking Benchmark (MGIT): Better Locating Target in Complex Spatio-temporal and Causal Relationship``, S. Hu, D. Zhang, M. Wu, X. Feng, X. Li, X. Zhao, K. Huang
        Thirty-seventh Conference on Neural Information Processing Systems Datasets and Benchmarks Track. 2023

    Args:
        root_dir (string): Root directory of dataset where ``train``,
            ``val`` and ``test`` folders exist.
        split (string, optional): Specify ``train``, ``val`` or ``test``
            subset of MGIT.
    """

    def __init__(self,  version='tiny'):
        super(MGIT, self).__init__()
        split = 'test'
        root_dir = '/home/local_data/benchmark/MGIT'
        assert split in ['train', 'val', 'test'], 'Unknown subset.'

        self.base_path = "/home/local_data/benchmark/MGIT"
        self.split = 'test'
        self.version = version  # temporarily, the toolkit only support tiny version of MGIT

        f = open(os.path.join(os.path.split(os.path.realpath(__file__))[0], 'mgit.json'), 'r', encoding='utf-8')
        self.infos = json.load(f)[self.version]
        f.close()

        self.sequence_list = self.infos[self.split]

        if split in ['train', 'val', 'test']:
            self.seq_dirs = [os.path.join(root_dir, 'data', split, s, 'frame_{}'.format(s)) for s in self.sequence_list]
            self.anno_files = [os.path.join(root_dir, 'attribute', 'groundtruth', '{}.txt'.format(s)) for s in
                               self.sequence_list]
            self.restart_files = [os.path.join(root_dir, 'attribute', 'restart', '{}.txt'.format(s)) for s in
                                  self.sequence_list]

    def __getitem__(self, index):
        r"""
        Args:
            index (integer or string): Index or name of a sequence.

        Returns:
            tuple:
                (img_files, anno, restart_flag), where ``img_files`` is a list of
                file names, ``anno`` is a N x 4 (rectangles) numpy array, while
                ``restart_flag`` is a list of
                restart frames.
        """
        if isinstance(index, six.string_types):
            if not index in self.sequence_list:
                raise Exception('Sequence {} not found.'.format(index))
            index = self.sequence_list.index(index)

        img_files = sorted(glob.glob(os.path.join(
            self.seq_dirs[index], '*.jpg')))

        anno = np.loadtxt(self.anno_files[index], delimiter=',')

        nlp_path = './mgit/datasets/mgit_nlp/{}.xlsx'.format(
            self.sequence_list[index])
        nlp_tab = pd.read_excel(nlp_path)
        nlp_rect = nlp_tab.iloc[:, [14]].values
        nlp_rect = nlp_rect[-1, 0]

        restart_flag = np.loadtxt(self.restart_files[index], delimiter=',', dtype=int)

        return img_files, anno, nlp_rect, restart_flag

    def __len__(self):
        return len(self.sequence_list)

    def get_sequence_list(self):
        return self.sequence_list