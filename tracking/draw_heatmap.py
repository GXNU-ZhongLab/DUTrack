import cv2
import numpy as np
import sys
import os

import torch


def vis_mask_token(heat_data, img=None, show_size=(150, 150), factor=0.4, window="feature"):
    """ 可视化特征

    Args:
        heat_data (_type_): (H, W)
        img (_type_, optional): _description_. Defaults to None.
        show_size (tuple, optional): _description_. Defaults to (150, 150).
        factor (float, optional): _description_. Defaults to 0.4.
        window (str, optional): _description_. Defaults to "feature".

    Returns:
        _type_: _description_
    """
    heat_data = heat_data.cpu().numpy()
    heat_data = cv2.resize(heat_data, show_size)

    heat_data_x = heat_data
    Min = np.min(heat_data_x)
    Max = np.max(heat_data_x)
    Sum = np.mean(heat_data_x)

    # sys.float_info.epsilon：是一个极小的数，用于避免除数为0的情况，即 heat_data矩阵为0的情况
    # heat_data_max = (heat_data_x - Min) / (Max - Min + sys.float_info.epsilon)
    if (Max - Min) != 0 and not np.isnan(Max - Min):
        heat_data_max = (heat_data_x - Min) / (Max - Min)
    else:
        heat_data_max = (heat_data_x - Min) / (Max - Min + sys.float_info.epsilon)

    heat_data = heat_data_max

    heat_data = np.uint8(255 * heat_data)
    heat_data = cv2.applyColorMap(heat_data, cv2.COLORMAP_JET)

    if img is not None:
        img = cv2.resize(img, show_size)
        heat_map_data = np.uint8(img * (1 - factor) + heat_data * factor)
    else:
        heat_map_data = heat_data

    font = cv2.FONT_HERSHEY_SIMPLEX
    heat_map_data = cv2.putText(heat_map_data, window, (0, 0), color=(255, 0, 0), fontFace=font, fontScale=1.2)
    return heat_map_data

def visualize_attn(attn, img,dataset_name,frame_num):
    '''
    img: (3,256,256)
    attn: (1,8,1,256)
    '''
    # print(attn)
    """
    将attn拆分为8个部分，每个部分生成一张彩色图片并保存到本地文件。
    img: 3x256x256的张量，表示原始图片。
    attn: 8x1x256的张量，表示注意力图。
    """
    img = cv2.cvtColor(img,cv2.COLOR_RGB2BGR)
    attn = attn[0].squeeze(0)
    attn = torch.mean(attn,dim=0)
    #print(attn.shape, img.shape, 'decoder_visualize')
    attn = attn[-256:,]
    heatmap_data =  vis_mask_token(attn.reshape(16,16), img)

    path = '/home/local_data/lxh/data/VLresult/Visual_attn/%s/%s.jpg'%(dataset_name,frame_num)
    if not os.path.exists(os.path.dirname(path)):
        os.makedirs(os.path.dirname(path))
    cv2.imwrite(path, heatmap_data)#如果路径不存在，保存失败，但是不会报错
