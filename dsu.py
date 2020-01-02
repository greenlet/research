import json
import os
import cv2
from nibabel import quaternions as quat
import numpy as np
from stl.mesh import Mesh


def get_pcn_from_mesh(mesh):
    eps = 1e-6
    pcn = []
    for ind in range(len(mesh.vectors)):
        vs = mesh.vectors[ind]
        n = mesh.normals[ind]
        nn = np.linalg.norm(n)
        if nn < eps:
            # print('Warning normal', n, vs)
            continue
        prod = 0
        for i in range(3):
            dv = vs[(i + 1) % 3, :] - vs[i, :]
            dvn = np.linalg.norm(dv)
            if dvn < eps:
                # print('Warning vectors', vs)
                break
            prod += np.abs(np.dot(dv, n) / (dvn * nn))
        if prod < eps:
            pcn.append(np.average(vs, axis=0))
    return np.reshape(pcn, (-1, 6))


def pose_dict_to_arr(pose):
    pm, qm = pose['p'], pose['q']
    p = np.array([pm['x'], pm['y'], pm['z']])
    q = np.array([qm['qw'], qm['qx'], qm['qy'], qm['qz']])
    rot = quat.quat2mat(q)
    return p, q, rot


def calc_normals(image_points):
    def ind_is_valid(ind):
        return ind[0] > 0 and ind[0] < image_points.shape[0] \
               and ind[1] > 0 and ind[1] < image_points.shape[1]

    def len_is_valid(l):
        return l > 0.001 and l < 0.5

    def get_normal(ind0, shift1, shift2):
        ind1 = ind0[0] + shift1[0], ind0[1] + shift1[1]
        ind2 = ind0[0] + shift2[0], ind0[1] + shift2[1]
        if not (ind_is_valid(ind1) and ind_is_valid(ind2)):
            return None
        p0 = image_points[ind0]
        p1 = image_points[ind1]
        p2 = image_points[ind2]
        v1, v2 = p1 - p0, p2 - p0
        l1, l2 = np.linalg.norm(v1), np.linalg.norm(v2)
        if not (len_is_valid(l1) and len_is_valid(l2)):
            return None
        n = np.cross(v1, v2) / (l1 * l2)
        ln = np.linalg.norm(n)
        if ln < 0.0001:
            return None
        return n / ln

    def calc_avg(normals):
        res = np.zeros((1, 3), np.float64)
        cnt = 0
        for normal in normals:
            if normal is not None:
                res += normal
                cnt += 1
        if cnt > 0:
            res /= cnt
        return res

    res = np.zeros(image_points.shape, image_points.dtype)
    height, width, _ = image_points.shape
    ind_shifts = [
        (-1, -1),
        (-1, 0),
        (-1, 1),
        (0, 1),
        (1, 1),
        (1, 0),
        (1, -1),
        (0, -1),
    ]
    ind_shifts.reverse()
    n_shifts = len(ind_shifts)
    for h in range(height):
        for w in range(width):
            normals = []
            for i in range(n_shifts):
                shift1 = ind_shifts[i]
                shift2 = ind_shifts[(i + 1) % n_shifts]
                normals.append(get_normal((h, w), shift1, shift2))
            res[h, w, :] = calc_avg(normals)
    return res


def calc_pcn_from_depth(depth, cam_int, cam_p, cam_rot, resize_factor=2):
    h, w = depth.shape
    y, x = np.array(range(h), np.float32), np.array(range(w), np.float32)
    y = (y * resize_factor - cam_int['cy']) / cam_int['fy']
    x = (x * resize_factor - cam_int['cx']) / cam_int['fy']
    pc = np.ones((h, w, 3), np.float32)
    pc[:, :, 0] = np.repeat(x[np.newaxis, :], h, axis=0)
    pc[:, :, 1] = np.repeat(y[:, np.newaxis], w, axis=1)
    pc[:, :, 2] = depth
    pc[:, :, 0] *= depth
    pc[:, :, 1] *= depth
    pc = np.dot(pc, cam_rot.T) + cam_p

    normals = calc_normals(pc)
    pcn = np.dstack((pc, normals))

    return pcn


def iterate_ds(pred_path, gt_path):
    raw_path = os.path.join(pred_path, 'raw')
    bts_data_path = os.path.join(gt_path, 'bts', 'data')
    file_names = os.listdir(raw_path)
    for file_name in file_names:
        file_name_base, _ = os.path.splitext(file_name)
        _, cam_id, _, id1, id2 = file_name_base.split('_')

        img_src_path = os.path.join(gt_path, f'cam_{cam_id}', file_name)
        img_src = cv2.imread(img_src_path)
        img_src = cv2.cvtColor(img_src, cv2.COLOR_BGR2RGB)

        depth_path = os.path.join(raw_path, file_name)
        depth_pred = cv2.imread(depth_path, -1)
        depth_pred = depth_pred.astype(np.float32) / 1000.0
        resize_factor = 2
        # resize_factor = 4
        # depth = depth[::2, ::2]

        depth_gt_name = file_name.replace('_img_', '_depth_map_')
        depth_gt_path = os.path.join(gt_path, 'bts', 'data', depth_gt_name)
        depth_gt = cv2.imread(depth_gt_path, -1)
        depth_gt = depth_gt.astype(np.float32) / 1000.0

        depth_gt_src_name = depth_gt_name.replace('.png', '.ext')
        depth_gt_src_path = os.path.join(gt_path, f'cam_{cam_id}', depth_gt_src_name)
        depth_gt_src = np.fromfile(depth_gt_src_path, np.float32) / 100
        depth_gt_src = np.reshape(depth_gt_src, (depth_gt.shape[0] * 2, depth_gt.shape[1] * 2))

        meta_name = f'meta_{id1}_{id2}.json'
        meta_path = os.path.join(gt_path, 'meta', meta_name)
        meta = json.load(open(meta_path, 'r'))

        cam = meta['cams'][cam_id]
        cam_int, cam_ext = cam['intrinsics'], cam['extrinsics']
        cam_p, _, cam_rot = pose_dict_to_arr(cam_ext)

        pose_gt = []
        for part_pose in meta['parts'].values():
            pose_gt.append(pose_dict_to_arr(part_pose))

        yield img_src, depth_pred, depth_gt, depth_gt_src, (cam_int, cam_p, cam_rot), pose_gt


