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


def iterate_predicted(pred_path, gt_path):
    raw_path = os.path.join(pred_path, 'raw')
    depth_names = os.listdir(raw_path)
    for depth_name in depth_names:
        depth_path = os.path.join(raw_path, depth_name)
        depth = cv2.imread(depth_path, -1)
        depth = depth.astype(np.float32) / 1000.0
        resize_factor = 2
        # resize_factor = 4
        # depth = depth[::2, ::2]

        depth_name_base, _ = os.path.splitext(depth_name)
        _, cam_id, _, id1, id2 = depth_name_base.split('_')
        meta_name = f'meta_{id1}_{id2}.json'
        meta_path = os.path.join(gt_path, 'meta', meta_name)
        meta = json.load(open(meta_path, 'r'))

        cam = meta['cams'][cam_id]
        h, w = depth.shape
        y, x = np.array(range(h), np.float32), np.array(range(w), np.float32)
        c_int, c_ext = cam['intrinsics'], cam['extrinsics']
        cam_p, _, cam_rot = pose_dict_to_arr(c_ext)
        y = (y * resize_factor - c_int['cy']) / c_int['fy']
        x = (x * resize_factor - c_int['cx']) / c_int['fy']
        pc = np.ones((h, w, 3), np.float32)
        pc[:, :, 0] = np.repeat(x[np.newaxis, :], h, axis=0)
        pc[:, :, 1] = np.repeat(y[:, np.newaxis], w, axis=1)
        pc[:, :, 2] = depth / np.sqrt(1 + pc[:, :, 0]**2 + pc[:, :, 1]**2)
        pc[:, :, 0] *= pc[:, :, 2]
        pc[:, :, 1] *= pc[:, :, 2]
        pc = np.dot(pc, cam_rot.T) + cam_p
        pc = np.reshape(pc, (-1, 3))

        pose_gt = []
        for part_pose in meta['parts'].values():
            pose_gt.append(pose_dict_to_arr(part_pose))

        yield pc, (cam_p, cam_rot), pose_gt


def test_ppf(stl_path, pred_path, gt_path):
    model_mesh = Mesh.from_file(stl_path)
    model_pcn = get_pcn_from_mesh(model_mesh)
    detector = cv2.ppf_match_3d_PPF3DDetector(0.025, 0.05)
    detector.trainModel(model_pcn)
    for scene_pc, cam_pos, pose_gt in iterate_predicted(pred_path, gt_path):
        cam_p, cam_rot = cam_pos
        viewpoint = tuple(cam_p)
        _, scene_pcn = cv2.ppf_match_3d.computeNormalsPC3d(scene_pc, 4, False, viewpoint)
        matches = detector.match(scene_pcn, 1.0 / 40.0, 0.15)
        print('---- Predicted pose:')
        for m in matches:
            print(m.pose)
        print('---- GT pose:')
        for p, q, rot in pose_gt:
            print(p, rot)


if __name__ == '__main__':
    stl_path = '/home/burakov/prog/tra/resources/parts/front_left_node_1/model.stl'
    pred_path = '/home/burakov/prog/depth/bts/result_bts_tra_2/'
    gt_path = '/home/burakov/Alpha/Data/Arrival/CORNER_NODE_LH_1000093_E_CORNER_NODE_RH_1000094_D_on_agv_000500/'
    test_ppf(stl_path, pred_path, gt_path)


