from lib.test.evaluation.environment import EnvSettings

def local_env_settings():
    settings = EnvSettings()

    # Set your local paths here.

    settings.davis_dir = ''
    settings.got10k_lmdb_path = '/home/local_data/benchmark/got10k_lmdb'
    settings.got10k_path = '/home/local_data/benchmark/got10k'
    settings.got_packed_results_path = ''
    settings.got_reports_path = ''
    settings.itb_path = '/home/local_data/benchmark/itb'
    settings.lasot_extension_subset_path = '/home/local_data/benchmark/lasot_extension_subset'
    settings.lasot_lmdb_path = '/home/local_data/benchmark/lasot_lmdb'
    settings.lasot_path = '/home/local_data/benchmark/lasot'
    settings.mgit_path = '/home/local_data/benchmark/MGIT'
    settings.network_path = '/home/local_data/lxh/history_work/DUTrack/output/test/networks'    # Where tracking networks are stored.
    settings.nfs_path = '/home/local_data/benchmark/nfs'
    settings.otb_path = '/home/local_data/benchmark/otb'
    settings.prj_dir = '/home/local_data/lxh/history_work/DUTrack'
    settings.result_plot_path = '/home/local_data/lxh/history_work/DUTrack/output/test/result_plots'
    settings.results_path = '/home/local_data/lxh/history_work/DUTrack/output/test/tracking_results'    # Where to store tracking results
    settings.save_dir = '/home/local_data/lxh/history_work/DUTrack/output'
    settings.segmentation_path = '/home/local_data/lxh/history_work/DUTrack/output/test/segmentation_results'
    settings.tc128_path = '/home/local_data/benchmark/TC128'
    settings.tn_packed_results_path = ''
    settings.tnl2k_path = '/home/local_data/benchmark/TNL2K'
    settings.otb_lang_path = '/home/local_data/benchmark/OTB_sentences'
    settings.tpl_path = ''
    settings.trackingnet_path = '/home/local_data/benchmark/trackingnet'
    settings.uav_path = '/home/local_data/benchmark/uav'
    settings.vot18_path = '/home/local_data/benchmark/vot2018'
    settings.vot22_path = '/home/local_data/benchmark/vot2022'
    settings.vot_path = '/home/local_data/benchmark/VOT2019'
    settings.youtubevos_dir = ''

    return settings

