import torch
# from mover.pose_optimization_cooperative import \
#     compute_optimal_translation_cam_inc_tensor


def compute_optimal_translation_cam_inc_tensor(bbox_target, vertices, cam_intrinc_mat=None, cam_extrinc_mat=None):
    """
    Computes the optimal translation to align the mesh to a bounding box using
    least squares.

    Args:
        bbox_target (tensor B x 4): bounding box in B, xywh.
        vertices (B x V x 3): Batched vertices.
        f (float): Focal length.
        cam_intrinc_mat (1 x 3 x 3)
        cam_extrinc_mat (1 x 3 x 3)
        img_size (int): Image size in pixels.

    Returns:
        Optimal 3D translation (B x 3).
    """

    # import pdb;pdb.set_trace()
    assert cam_intrinc_mat is not None
    f = cam_intrinc_mat[0][0][0]

    bbox_mask = bbox_target
    mask_center = bbox_mask[:, :2] + bbox_mask[:, 2:] / 2
    diag_mask = (bbox_mask[:, 2] ** 2 + bbox_mask[:, 3] ** 2).sqrt()
    

    B = vertices.shape[0]
    x = torch.zeros(B).cuda()
    y = torch.zeros(B).cuda()
    z = 0.4 * torch.ones(B).cuda() #0.4

    # import pdb;pdb.set_trace()
    for _ in range(50):
        translation = torch.stack((x, y, z), -1).unsqueeze(1) # B, 1, 3
        v = vertices + translation
        
        bbox_proj = compute_bbox_proj_cam_inc(v, cam_intrinc_mat, cam_extrinc_mat)
        # print(f'Update vertices: {_}', v[0,0,:], translation[0])
        # print(f'bbox proj', bbox_proj[0])

        if False:
            # debug
            import cv2
            cv2_img = cv2.imread('/is/cluster/work/hyi/results/HDSR/PROX_qualitative/N3OpenArea_00157_01/Total3D/recon.png')
            cv2_img = cv2.resize(cv2_img, (640, 360))
            
            for one in bbox_proj:
                xmin, ymin, w, h = one
                xmin = int(xmin)
                ymin = int(ymin)
                w = int(w) 
                h = int(h)
                xmax = xmin + w
                ymax = ymin + h
                img = cv2.line(cv2_img, (xmin,ymin), (xmin,ymax), color=(255,0,0), thickness=2)
                img = cv2.line(cv2_img,(xmin,ymin), (xmax,ymin), color=(255,0,0),thickness=2)
                img = cv2.line(cv2_img, (xmax,ymin), (xmax,ymax), color=(255,0,0),thickness=2)
                img = cv2.line(cv2_img,(xmin,ymax), (xmax,ymax), color=(255,0,0),thickness=2)
            xmin, ymin, w, h = bbox_target[0]
            xmin = int(xmin)
            ymin = int(ymin)
            w = int(w) 
            h = int(h)
            xmax = xmin + w
            ymax = ymin + h
            img = cv2.line(cv2_img, (xmin,ymin), (xmin,ymax), color=(0,0,255), thickness=2)
            img = cv2.line(cv2_img,(xmin,ymin), (xmax,ymin), color=(0,0,255),thickness=2)
            img = cv2.line(cv2_img, (xmax,ymin), (xmax,ymax), color=(0,0,255),thickness=2)
            img = cv2.line(cv2_img,(xmin,ymax), (xmax,ymax), color=(0,0,255),thickness=2)
            
            cv2.imwrite(os.path.join('/is/cluster/hyi/workspace/HCI/hdsr/mover_ori_repo_total3D/hdsr/mover_ori_repo/debug/cal_bbox', f'{_}.jpg'), cv2_img)
            # cv2.imshow("demo", cv2_img)
            # cv2.waitKey(500)
        
        diag_proj = torch.sqrt(torch.sum(bbox_proj[:, 2:] ** 2, 1))

        delta_z = z * (diag_proj / diag_mask - 1)
        
        z = z + delta_z
        # change z will leads to the xy change;
        proj_center = bbox_proj[:, :2] + bbox_proj[:, 2:] / 2
        x += (mask_center[:, 0] - proj_center[:, 0]) * z / f
        y += (mask_center[:, 1] - proj_center[:, 1]) * z / f 
        
    return torch.stack((x, y, z), -1)

def get_init_translation(self, samples=20):
    bbox_xywh = torch.stack((self.det_results[:, 0], self.det_results[:, 1], 
                        self.det_results[:, 2]-self.det_results[:, 0],
                        self.det_results[:, 3]-self.det_results[:, 1]), 1)
    
    verts_object_world_gp = self.get_verts_object_parallel_ground()
    verts_object_world_gp = verts_object_world_gp.reshape(samples, -1, 3) 
    camera_extrin_mat = self.get_cam_extrin()
    camera_intrin_mat = self.K_intrin
    translation_init = compute_optimal_translation_cam_inc_tensor(bbox_xywh, verts_object_world_gp, \
        camera_intrin_mat, camera_extrin_mat)

    # set translation
    self.translations_object.data = translation_init + self.translations_object.data
    
    print(f'calculate translation in get_init_translation, \
        translations_object requires_grad:{self.translations_object.requires_grad}')

    