import numpy as np
import torch
import itertools

import cv2
from smplx.lbs import transform_mat

##### 500 pixels 
LARGE_ERROR = 5e2

def multiview_weight_adjustment(keypoints, 
                    cameras, 
                    sigma=50, 
                    weighted_joints_idxs = [3, 4, 6, 7, 
                                            10, 11, 13, 14,
                                            19, 20, 21, 22, 23, 24]
                    ):

    projs = list()
    for cam in cameras:
        camera_transform = transform_mat(cam.rotation, cam.translation.unsqueeze(dim=-1)).detach().cpu()
        intrisic = torch.tensor([[cam.focal_length_x, 0, cam.center[0, 0]], 
                                 [0, cam.focal_length_y, cam.center[0, 1]], 
                                 [0, 0, 1]], dtype=cam.dtype)

        projs.append(intrisic @ camera_transform[0,:3])

    pair_list = list(itertools.combinations(range(len(cameras)), 2))

    for j in weighted_joints_idxs:
        reproj_error = [.0] * len(cameras)
        conf_j = [.0] * len(cameras)
        visit_count = [0] * len(cameras)
        # given a view pair, create a 3D proposal
        for u, v in pair_list: 

            joint_u = keypoints[u][0, j, :2].numpy().squeeze()
            conf_u = keypoints[u][0, j, 2]
            conf_j[u] = conf_u
            joint_v = keypoints[v][0, j, :2].numpy().squeeze()
            conf_v = keypoints[v][0, j, 2]
            conf_j[v] = conf_v


            if conf_u == 0:
                continue
            if conf_v == 0:
                continue

            proj_u = projs[u]
            proj_v = projs[v]

            joint_3D = torch.tensor(cv2.triangulatePoints(  proj_u.numpy(), proj_v.numpy(), 
                                                            joint_u, joint_v))
            joint_3D /= joint_3D[3]

            # project the proposal back to 2D (including u, v) to see the reproj error
            for i, proj in enumerate(projs):

                conf_i = keypoints[i][0, j, 2]
                conf_j[i] = conf_i

                if conf_i == 0:
                    continue

                joint_uv2i = proj @ joint_3D
                joint_uv2i /= joint_uv2i[2]

                diff = keypoints[i][0, j, :2] - joint_uv2i[:2].reshape(1, -1)
                reproj_error_uvi = torch.sqrt(torch.sum(diff**2))
                reproj_error[i] += reproj_error_uvi
                visit_count[i] += 1
                

        avg_reproj_error = [r/v if v != 0 else torch.tensor(LARGE_ERROR) 
                                for r, v in zip(reproj_error, visit_count)]
        w = torch.tensor([1 / (1 + torch.exp(d / sigma)) for d in avg_reproj_error])

        ## Baysian's rule
        w = w * torch.tensor(conf_j)

        # normalize w.r.t. to avg. of non-zero weights
        all_view_sum = sum(w)
        avg_visible = 0 if all_view_sum == 0 else ( all_view_sum/ len([e for e in w if e != 0.0]))
        w /= (avg_visible + np.finfo(np.float32).eps)

        for vid, w_v in enumerate(w):
            keypoints[vid][0, j, 2] = w_v
    
    return keypoints
