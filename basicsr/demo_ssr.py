# ------------------------------------------------------------------------
# Copyright (c) 2022 megvii-model. All Rights Reserved.
# ------------------------------------------------------------------------
# Modified from BasicSR (https://github.com/xinntao/BasicSR)
# Copyright 2018-2020 BasicSR Authors
# ------------------------------------------------------------------------
import torch
import os
# from basicsr.data import create_dataloader, create_dataset
from basicsr.models import create_model
from basicsr.utils import FileClient, imfrombytes, img2tensor, padding, tensor2img, imwrite, set_random_seed

import argparse
from basicsr.utils.options import dict2str, parse
from basicsr.utils.dist_util import get_dist_info, init_dist
import random

def parse_options(is_train=True):
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '-opt', type=str, required=True, help='Path to option YAML file.')
    parser.add_argument(
        '--launcher',
        choices=['none', 'pytorch', 'slurm'],
        default='none',
        help='job launcher')
    parser.add_argument('--local_rank', type=int, default=0)

    parser.add_argument('--img_pair_paths', type=str, required=True, help='文件夹路径，包含左右视图图像对。')
    parser.add_argument('--output_dir', type=str, required=True, help='Directory for output images.')

    args = parser.parse_args()
    opt = parse(args.opt, is_train=is_train)

    # distributed settings
    if args.launcher == 'none':
        opt['dist'] = False
        print('Disable distributed.', flush=True)
    else:
        opt['dist'] = True
        if args.launcher == 'slurm' and 'dist_params' in opt:
            init_dist(args.launcher, **opt['dist_params'])
        else:
            init_dist(args.launcher)
            print('init dist .. ', args.launcher)

    opt['rank'], opt['world_size'] = get_dist_info()

    # random seed
    seed = opt.get('manual_seed')
    if seed is None:
        seed = random.randint(1, 10000)
        opt['manual_seed'] = seed
    set_random_seed(seed + opt['rank'])

    opt['img_path'] = {
        'img_pair_paths': args.img_pair_paths,
        'output_dir': args.output_dir,
    }
    # print('opt', opt)
    # print(type(opt))
    return opt

def imread(img_path):
    file_client = FileClient('disk')
    img_bytes = file_client.get(img_path, None)
    try:
        img = imfrombytes(img_bytes, float32=True)
    except:
        raise Exception("path {} not working".format(img_path))

    img = img2tensor(img, bgr2rgb=True, float32=True)
    return img

def process_image_pair(img_l_path, img_r_path, output_dir, model):
    # 读取左右两张图像
    img_l = imread(img_l_path)
    # print('img_l', img_l_path)
    img_r = imread(img_r_path)
    # print('img_r', img_r_path)

    # 将两张图像拼接在一起
    img = torch.cat([img_l, img_r], dim=0)

    # 运行模型进行推理
    model.feed_data(data={'lq': img.unsqueeze(dim=0)})
    model.test()

    # 获取处理后的图像
    visuals = model.get_current_visuals()
    sr_img_l = visuals['result'][:, :3]
    sr_img_r = visuals['result'][:, 3:]

    # 将 Tensor 转换为图像
    sr_img_l, sr_img_r = tensor2img([sr_img_l, sr_img_r])

    # 生成输出文件的路径
    base_name_l = os.path.splitext(os.path.basename(img_l_path))[0]
    base_name_r = os.path.splitext(os.path.basename(img_r_path))[0]
    output_l_path = os.path.join(output_dir, f'{base_name_l}.png')
    output_r_path = os.path.join(output_dir, f'{base_name_r}.png')

    # 保存处理后的图像
    imwrite(sr_img_l, output_l_path)
    imwrite(sr_img_r, output_r_path)

    print(f'输出保存至 {output_l_path} 和 {output_r_path}')

def main():
    # parse options, set distributed setting, set ramdom seed
    opt = parse_options(is_train=False)
    opt['num_gpu'] = torch.cuda.device_count()

    img_pair_dir = opt['img_path'].get('img_pair_paths')
    output_dir = opt['img_path'].get('output_dir')

    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    ## 2. run inference
    opt['dist'] = False
    model = create_model(opt)

    image_pairs = []
    for img_name in sorted(os.listdir(img_pair_dir)):
        if '_L' in img_name:
            img_l_path = os.path.join(img_pair_dir, img_name)
            img_r_path = os.path.join(img_pair_dir, img_name.replace('_L', '_R'))
            image_pairs.append((img_l_path, img_r_path))
    for img_l_path, img_r_path in image_pairs:
        process_image_pair(img_l_path, img_r_path, output_dir, model)

if __name__ == '__main__':
    main()