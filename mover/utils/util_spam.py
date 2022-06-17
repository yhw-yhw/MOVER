import torch
import neural_renderer as nr

def get_faces_and_textures(verts_list, faces_list):
    """

    Args:
        verts_list (List[Tensor(B x V x 3)]).
        faces_list (List[Tensor(f x 3)]).

    Returns:
        faces: (1 x F x 3)
        textures: (1 x F x 1 x 1 x 1 x 3)
    """
    # TODO:
    colors_list = [
        [251 / 255.0, 128 / 255.0, 114 / 255.0],  # red
        [0.65098039, 0.74117647, 0.85882353],  # blue
        [0.9, 0.7, 0.7],  # pink
    ]

    all_faces_list = []
    all_textures_list = []
    o = 0
    for verts, faces, colors in zip(verts_list, faces_list, colors_list):
        B = len(verts)
        index_offset = torch.arange(B).to(verts.device) * verts.shape[1] + o
        index_offset = index_offset.int()
        o += verts.shape[1] * B
        faces_repeat = faces.clone().repeat(B, 1, 1)
        faces_repeat += index_offset.view(-1, 1, 1)
        faces_repeat = faces_repeat.reshape(-1, 3)
        all_faces_list.append(faces_repeat)
        textures = torch.FloatTensor(colors).to(verts.device)
        all_textures_list.append(textures.repeat(faces_repeat.shape[0], 1, 1, 1, 1))
        
    all_faces_list = torch.cat(all_faces_list).unsqueeze(0)
    all_textures_list = torch.cat(all_textures_list).unsqueeze(0)
    return all_faces_list, all_textures_list


def project_bbox(vertices, renderer, parts_labels=None, bbox_expansion=0.0):
    """
    Computes the 2D bounding box of the vertices after projected to the image plane.

    TODO(@jason): Batch these operations.

    Args:
        vertices (V x 3).
        renderer: Renderer used to get camera parameters.
        parts_labels (dict): Dictionary mapping a part name to the corresponding vertex
            indices.
        bbox_expansion (float): Amount to expand the bounding boxes.

    Returns:
        If a part_label dict is given, returns a dictionary mapping part name to bbox.
        Else, returns the projected 2D bounding box.
    """
    proj = nr.projection(
        (vertices * torch.tensor([[1, -1, 1.0]]).cuda()).unsqueeze(0),
        K=renderer.K,
        R=renderer.R,
        t=renderer.t,
        dist_coeffs=renderer.dist_coeffs,
        orig_size=1,
    )
    proj = proj.squeeze(0)[:, :2]
    if parts_labels is None:
        parts_labels = {"": torch.arange(len(vertices)).to(vertices.device)}
    bbox_parts = {}
    for part, inds in parts_labels.items():
        bbox = torch.cat((proj[inds].min(0).values, proj[inds].max(0).values), dim=0)
        if bbox_expansion:
            center = (bbox[:2] + bbox[2:]) / 2
            b = (bbox[2:] - bbox[:2]) / 2 * (1 + bbox_expansion)
            bbox = torch.cat((center - b, center + b))
        bbox_parts[part] = bbox
    if "" in parts_labels:
        return bbox_parts[""]
    return bbox_parts

# ! Warning : add body pose filter: on OP confidence, especially for multiple images which does not have consistency human motion. 
def get_ordinal_depth_wrong_list():
    det_result_dir = opt.img_dir_det
    img_list = opt.img_list
    perframe_det_bbox2D_list, perframe_masks_list, perframe_cam_rois_list = load_perframe_det_results(det_result_dir, img_list, opt.width, None, device, opt.USE_MASK)
    import pdb;pdb.set_trace()
    assert len(perframe_masks_list) == vertices_np.shape[0]
    model.set_perframe_det_result(perframe_masks=perframe_masks_list,
        perframe_det_results=perframe_det_bbox2D_list,
        perframe_cam_rois=perframe_cam_rois_list)
    ordinal_depth_loss_list = model(smplx_model_vertices=torch.from_numpy(vertices_np).to(device), 
                            stage=30) 
    import pickle
    with open(os.path.join(save_dir, 'st3_ordinal_depth_loss_perframe.pkl'), 'w') as fin:
        pickle.dump({'ordinal_depth_loss': ordinal_depth_loss_list}, fin)
    