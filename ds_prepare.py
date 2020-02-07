import json
import os
import random
import shutil
import numpy as np
from PIL import Image, ImageDraw
import utils


MAX_DEPTH = 30
BTS_SUBDIR = 'bts'
BTS_DATA_SUBDIR = 'data'


def iterate_metas(path):
    for meta_path in utils.list_files(path, paths_only=True):
        f = open(meta_path, 'r')
        yield json.load(f)


def iterate_files(ds_path):
    if not ds_path:
        return
    prefix = utils.prefixed(ds_path)
    meta_path = prefix('meta')
    for meta in iterate_metas(meta_path):
        for cam in meta['cams'].values():
            image_path = prefix(cam['img'])
            image_name = os.path.split(image_path)[-1]
            image = Image.open(image_path)
            depth_path = cam.get('depth_map_name')
            depth_name, depth = None, None
            if depth_path:
                depth_path = prefix(depth_path)
                depth_name = os.path.split(depth_path)[-1]
                depth = np.fromfile(depth_path, np.float32) / 100
                w, h = image.size
                depth = np.reshape(depth, (h, w))
            yield cam, image_path, depth_path, image_name, depth_name, image, depth


def read_vis(ds_path):
    out_path = utils.make_dir(ds_path, 'out_vis')
    for cam, image_path, depth_path, image_name, depth_name, image, depth in iterate_files(ds_path):
        depth_excess_inds = np.where(depth > MAX_DEPTH)
        if len(depth_excess_inds[0]):
            depth_excess = depth[depth_excess_inds]
            print(depth_name, depth_excess.min(), depth_excess.shape[0])
            draw = ImageDraw.Draw(image)

            for p in zip(*depth_excess_inds):
                cross_len = 6
                pu = (p[1] - cross_len, p[0] - cross_len)
                pd = (p[1] + cross_len, p[0] + cross_len)
                pl = (p[1] - cross_len, p[0] + cross_len)
                pr = (p[1] + cross_len, p[0] - cross_len)
                draw.line([pl, pr], fill='red', width=2)
                draw.line([pd, pu], fill='red', width=2)
            del draw
            file_out_path = os.path.join(out_path, image_name)
            resize_ratio = 1 / 2
            size = (int(dim * resize_ratio) for dim in image.size)
            image = image.resize(size, Image.BILINEAR)
            image.save(file_out_path)


def fix_depth(depth, p):
    max_sz = np.max(depth.shape)
    sz = 0
    while sz < max_sz:
        sz += 1
        submat = depth[p[0]-sz:p[0]+sz+1, p[1]-sz:p[1]+sz+1]
        vals = submat[submat < MAX_DEPTH]
        if len(vals):
            new_depth = vals.mean()
            # print('fix', p, depth[p], '-->', new_depth, sz)
            depth[p] = vals.mean()
            return
    print('Warning did not fixed', depth.shape, depth[p], p)


def make_train_test_ds(ds_path, resize_factor=2):
    bts_path = utils.make_dir(ds_path, BTS_SUBDIR)
    bts_data_path = utils.make_dir(ds_path, BTS_SUBDIR, BTS_DATA_SUBDIR)
    res = []
    for cam, image_path, depth_path, image_name, depth_name, image, depth in iterate_files(ds_path):
        print(image_name, depth_name)

        if resize_factor != 1:
            new_size = (image.size[0] // resize_factor, image.size[1] // resize_factor)
            # Resize and save image
            image = image.resize(new_size, Image.BILINEAR)

        bts_image_name = image_name
        bts_image_path = os.path.join(bts_data_path, bts_image_name)
        image.save(bts_image_path)

        bts_depth_name = 'none'
        if depth is not None:
            # Fix depth
            depth_excess_inds = np.where(depth > MAX_DEPTH)
            if len(depth_excess_inds[0]):
                for p in zip(*depth_excess_inds):
                    fix_depth(depth, p)
            depth *= 1000
            depth = depth.astype(np.int32)
            depth_i = Image.fromarray(depth, 'I')
            # Resize and save depth as png
            depth_i = depth_i.resize(new_size)
            bts_depth_name = os.path.splitext(depth_name)[0] + '.png'
            bts_depth_path = os.path.join(bts_data_path, bts_depth_name)
            depth_i.save(bts_depth_path)

        res.append((bts_image_name, bts_depth_name, cam['intrinsics']['fx'] / resize_factor))

    res_file_name = 'meta.txt'
    res_file_path = os.path.join(bts_path, res_file_name)
    with open(res_file_path, 'w') as fres:
        for item in res:
            fres.write('%s %s %.4f\n' % item)

    split_train_test(ds_path)


def split_train_test(ds_path):
    prefix = utils.prefixed(ds_path, BTS_SUBDIR)
    meta_path = prefix('meta.txt')
    with open(meta_path, 'r') as f:
        lines = f.readlines()
    random.shuffle(lines)
    train, test = lines[:-100], lines[-100:]
    train_path, test_path = prefix('train.txt'), prefix('test.txt')
    print(f'Splitting total {len(lines)} samples to {len(train)} train / {len(test)} test samples')
    with open(train_path, 'w') as f:
        f.writelines(train)
    with open(test_path, 'w') as f:
        f.writelines(test)


def calc_metrics(ds_path):
    file_paths = utils.list_files(ds_path, BTS_SUBDIR, BTS_DATA_SUBDIR)
    max_depth = 0
    for file_path, file_name in file_paths:
        depth_image = Image.open(file_path).convert('I')
        depth = np.asarray(depth_image, np.int32)
        depth = depth.astype(np.float32) / 1000
        m = depth.max()
        print(file_name, m)
        max_depth = max(max_depth, m)
    print('---- Max depth:', max_depth)


if __name__ == '__main__':
    # ds_path = '/media/burakov/Alpha/Data/CORNER_NODE_LH_1000093_E_CORNER_NODE_RH_1000094_D_on_agv_000500/'
    # ds_path = '/home/burakov/Alpha/Data/2Parts_assembling_left_000500/'
    # read_vis(ds_path)
    # ds_path = '/media/burakov/HardDrive/Data/CORNER_NODE_LH_1000093_E_CORNER_NODE_RH_1000094_D_on_agv_000500/'
    ds_path = '/media/burakov/HardDrive/Data/20200130_fslab1/'
    # ds_path = '/media/burakov/HardDrive/Data/2Parts_assembling_left_001000/'

    # read_vis(ds_path)
    make_train_test_ds(ds_path, 2)
    # split_train_test(ds_path)
    # calc_metrics(ds_path)

