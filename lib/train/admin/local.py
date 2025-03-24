class EnvironmentSettings:
    def __init__(self):
        self.workspace_dir = '/home/local_data/lxh/history_work/DUTrack'    # Base directory for saving network checkpoints.
        self.tensorboard_dir = '/home/local_data/lxh/history_work/DUTrack/tensorboard'    # Directory for tensorboard files.
        self.pretrained_networks = '/home/local_data/lxh/history_work/DUTrack/pretrained_networks'
        self.lasot_dir = '/home/local_data/benchmark/lasot'
        self.got10k_dir = '/home/local_data/benchmark/got10k/train'
        self.got10k_val_dir = '/home/local_data/benchmark/got10k/val'
        self.lasot_lmdb_dir = '/home/local_data/benchmark/lasot_lmdb'
        self.got10k_lmdb_dir = '/home/local_data/benchmark/got10k_lmdb'
        self.trackingnet_dir = '/home/local_data/benchmark/trackingnet'
        self.trackingnet_lmdb_dir = '/home/local_data/benchmark/trackingnet_lmdb'
        self.coco_dir = '/home/local_data/benchmark/coco'
        self.coco_lmdb_dir = '/home/local_data/benchmark/coco_lmdb'
        self.lvis_dir = ''
        self.sbd_dir = ''
        self.imagenet_dir = '/home/local_data/benchmark/vid'
        self.imagenet_lmdb_dir = '/home/local_data/benchmark/vid_lmdb'
        self.imagenetdet_dir = ''
        self.ecssd_dir = ''
        self.hkuis_dir = ''
        self.msra10k_dir = ''
        self.tnl2k_dir = '/home/local_data/benchmark/TNL2K'
        self.mgit_dir = '/home/local_data/benchmark/MGIT'
