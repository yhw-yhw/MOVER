import cv2
import matplotlib.pyplot as plt
import numpy as np
import tqdm
import torch
from PIL import Image
import sys
import os
import trimesh
import joblib
import glob
import json
import math
from loguru import logger
import pickle

from thirdparty.body_models.smplifyx.utils_mics.utils import get_image
from thirdparty.body_models.video_smplifyx.tf_utils import save_scalars, save_images

from mover.utils import write_opencv_matrix
from mover.dataset import *
from mover.utils import (bbox_xy_to_wh, center_vertices, \
    get_instance_masks_within_global_cam, write_opencv_matrix, get_local_mask_with_roi_camera)
from mover.constants import (
                RENDER_DEBUG, SAVE_ALL_VIEW_IMG, DEBUG_OBJ_SIZE, \
                SAVE_SCENE_RENDER_IMG, SAVE_APARENT_IMG, KPTS_OPENPOSE_FOR_SMPLX, \
                RENDER_SCAN,
                DEFAULT_LOSS_WEIGHTS,
                IMAGE_SIZE,
                TB_DEBUG,
                DEBUG_LOSS,
                DEBUG_FILTER_POSE,
                EVALUATE,
                USE_FILTER_CONF,
                SAVE_BEST_MODEL,
                SAVE_ALL_RENDER_RESULT,
                SAVE_DEPTH_MAP,
                DEBUG_LOSS_OUTPUT,
)

def load_smplify_result_without_scene(pare_dir, opt, device, model=None, img_fn_list=None, save_dir=None, render=False):                          
    pkl_fn = glob(os.path.join(pare_dir, 'results', '*.pkl'))
    all_obj_list = []
    img_list = opt.img_list
    
    if len(pkl_fn) > 0 and 'all' in pkl_fn[0]: # load 001_all.pkl
        pre_smplx_model = joblib.load(pkl_fn[0])
        new_pre_smplx_model = {}
        
        if type(pre_smplx_model) == dict:
            batch_size = opt.batch_size
            for key, val in pre_smplx_model.items():
                new_pre_smplx_model[key] = val[:batch_size]
    else:
        pkl_fn = glob(os.path.join(pare_dir, 'results/split', '*.pkl'))
        batch_size = opt.batch_size
        new_pre_smplx_model = {}

        # split parameters
        for one in range(batch_size):
            pre_smplx_model = joblib.load(pkl_fn[one])
            for key, val in pre_smplx_model.items():
                if key not in new_pre_smplx_model:
                    new_pre_smplx_model[key] = [val]
                else:
                    new_pre_smplx_model[key].append(val)
        
        for key, val in new_pre_smplx_model.items():
            new_pre_smplx_model[key] = np.stack(val)
            
    # load object vertices 
    mesh_dir = os.path.join(pare_dir, 'meshes')
    vertices_list = []
    for idx in tqdm(range(len(img_list)), desc='load pare'):
        obj_fn = os.path.join(mesh_dir, f'{img_list[idx]:03d}.obj')
        if not os.path.exists(obj_fn):
            obj_fn = os.path.join(mesh_dir, f'{img_list[idx]:06d}.obj')
            if not os.path.exists(obj_fn): #! for PROXD
                obj_fn = os.path.join(mesh_dir, f'{img_list[idx]:06d}.ply')

        all_obj_list.append(obj_fn)
        mesh = trimesh.load(obj_fn, process=False)
        vertices = np.asarray(mesh.vertices, dtype=np.float32)
        st_2_body_vertices = torch.from_numpy(vertices).to(device)
        r_vertices = st_2_body_vertices
        vertices_list.append(r_vertices.detach().cpu().numpy())

        # visualize
        if render and save_dir is not None:
            assert model is not None
            model.verts_person_og = r_vertices
            one_img = get_image(img_fn_list[idx])
            render_model_to_imgs_pyrender(model, one_img, False, \
                            f'st2_scene_human_init_{img_list[idx]:06d}', save_dir=os.path.join(save_dir, 'smplifyx'), scanned_scene=None)
    
    vertices_np = np.stack(vertices_list)
    
    return vertices_np, new_pre_smplx_model, all_obj_list

def add_tb_message(tb_logger, losses, loss_dict, loss_dict_weighted, all_iters):
    obj_num = len(losses)
    for sample_idx in range(obj_num):
        tmp_loss_dict = {}
        tmp_loss_dict_w = {}
        for t_k, t_v in loss_dict.items():
            tmp_loss_dict[t_k] = t_v[sample_idx]
            if t_k in loss_dict_weighted:
                tmp_loss_dict_w[t_k] = loss_dict_weighted[t_k][sample_idx]
        tmp_loss_dict_w['total_loss'] = losses[sample_idx]
        save_scalars(tb_logger, f'loss_{sample_idx}', tmp_loss_dict, all_iters)
        save_scalars(tb_logger, f'loss_weight_{sample_idx}', tmp_loss_dict_w, all_iters)


def get_depth_map(model, vertices_np_scene, img_list, template_save_dir, device):
    # get depth range map and save it.
    smplx_model_vertices =torch.from_numpy(vertices_np_scene).to(device)
    smplx_model_vertices = model.get_person_wrt_world_coordinates(smplx_model_vertices)
    # smplx_model_vertices is in world_CS
    smplx_model_vertices_cam = torch.transpose(torch.matmul(model.get_cam_extrin(), \
                torch.transpose(smplx_model_vertices, 2, 1)), 2, 1) 
    model.accumulate_ordinal_depth_from_human(smplx_model_vertices_cam, img_list=img_list, \
        debug=DEBUG_LOSS_OUTPUT, save_dir=template_save_dir)

def load_mesh_obj(mesh_fn=None):
    if mesh_fn is None:
        mesh_fn = os.path.join(os.path.dirname(__file__), '../data/body_example/000.obj')
    mesh = trimesh.load(mesh_fn, process=False)
    vertices = np.asarray(mesh.vertices, dtype=np.float32)
    faces = np.asarray(mesh.faces, dtype=np.int32)
    
    return vertices.squeeze(), faces.squeeze()
    
def load_img_Image(img_path, scale):
    image_pil = Image.open(img_path)
    image_pil = image_pil.resize(scale)
    image = np.array(image_pil)
    return image

def load_all_humans(opt, pare_dir, tmp_save_dir, save_dir, device=torch.device('cuda')):
    
    tmp_verts_path = os.path.join(tmp_save_dir, f'verts_input.pickle')
    if not opt.preload_body or not os.path.exists(tmp_verts_path):
        vertices_np, new_pre_smplx_model, all_obj_list = load_smplify_result_without_scene(pare_dir, 
            opt, device, save_dir=None, render=False) #render pose results on image.
        logger.info('load body verts')
        with open(os.path.join(tmp_save_dir, f'verts_input.pickle'), 'wb') as fout:
            pickle.dump( 
                {'vertices_np': vertices_np,
                'new_pre_smplx_model': new_pre_smplx_model,
                'all_obj_list': all_obj_list},fout)
    else:
        logger.info('preload body verts')
        with open(tmp_verts_path, 'rb') as fin:
            verts_input_dict = pickle.load(fin)
        vertices_np = verts_input_dict['vertices_np']
        new_pre_smplx_model = verts_input_dict['new_pre_smplx_model']
        all_obj_list = verts_input_dict['all_obj_list']

    return vertices_np, new_pre_smplx_model, all_obj_list

def recalculate_segment(ori_list, new_fps, principle_fps=30):
    if new_fps == principle_fps:
        return ori_list
    scale = new_fps / principle_fps
    return [int(one * scale) for one in ori_list]
    
def get_filter_human_poses(tmp_save_dir, stage3_dict, opt, st_2_fit_body_use_dict, \
                vertices_np, new_pre_smplx_model, save_dir, img_list, tb_logger, op_filter_flag, \
                random_sample_order,
                random_start_f,
                original_batch_size,
                all_obj_list, 
                posa_dir,
                st2_render_body_dir,
                img_fn_list
                ):
    
    filter_open = stage3_dict['filter_open']
    video_process = stage3_dict['video_process']
    segments_list = stage3_dict['segments_list']
    input_fps = opt.input_fps
    fps_list = recalculate_segment(stage3_dict['fps_list'], input_fps) 
    
    ####################################
    ### video segmentations
    ####################################
    total_seg_i = 0
    num_segments = segments_list[total_seg_i]
    per_segment_fps = fps_list[total_seg_i]
    perframes = math.ceil(vertices_np.shape[0] / num_segments)
    s_i = 0
    start_frame = s_i * perframes
    end_frame = min((s_i + 1) * perframes, vertices_np.shape[0])
    ## end of optimization settings
    
    tmp_verts_path = os.path.join(tmp_save_dir, f'verts_filter.pickle')
    if not opt.preload_body or not os.path.exists(tmp_verts_path): # always run.
        logger.info('recalculate filter body verts')
        # body filter: shape is bxnx3
        vertices_np_scene, body2scene_conf, filter_flag, ori_body2scene_conf = filter_human_poses(\
            filter_open, video_process,
            st_2_fit_body_use_dict, vertices_np, 
            per_segment_fps, start_frame, end_frame, new_pre_smplx_model, save_dir, s_i, num_segments,
            img_list, TB_DEBUG, tb_logger,
            use_filter_conf=USE_FILTER_CONF, model=None, op_filter=op_filter_flag)
        logger.info('load body verts')

        # if random_sample_order, start_frame, all_frame: use all videos into filtered method, 
        # and filter out from start_frame-batch_size.
        if random_sample_order != -1:

            logger.info(f'filter verts between : {random_start_f} - {random_start_f+original_batch_size}')
            
            tmp_filter_list = []
            cnt = 0
            for tmp_i in range(len(img_list)):
                if filter_flag[tmp_i]:
                    if tmp_i >= random_start_f and tmp_i < (random_start_f + original_batch_size): 
                        tmp_filter_list.append(cnt)
                    else:
                        filter_flag[tmp_i] = False
                    cnt += 1
            assert filter_flag.sum() == len(tmp_filter_list)
            vertices_np_scene = vertices_np_scene[tmp_filter_list]
            body2scene_conf = body2scene_conf[tmp_filter_list]
            ori_body2scene_conf = ori_body2scene_conf[tmp_filter_list]
            logger.info(f'random sample filter out: {len(tmp_filter_list)} from {cnt}')

        
        with open(os.path.join(tmp_save_dir, f'verts_filter.pickle'), 'wb') as fout:
            pickle.dump( 
                {'vertices_np_scene': vertices_np_scene,
                'body2scene_conf': body2scene_conf,
                'filter_flag': filter_flag,
                'ori_body2scene_conf': ori_body2scene_conf},fout)

    else:
        logger.info('preload  filter body verts')
        with open(tmp_verts_path, 'rb') as fin:
            verts_input_dict = pickle.load(fin)
        vertices_np_scene = verts_input_dict['vertices_np_scene']
        body2scene_conf = verts_input_dict['body2scene_conf']
        filter_flag = verts_input_dict['filter_flag']
        ori_body2scene_conf = verts_input_dict['ori_body2scene_conf']
    
    if DEBUG_FILTER_POSE:
        save_filter_render_body_results(all_obj_list, filter_flag, save_dir, st2_render_body_dir)

    # get filtered obj list and contact label list
    filter_obj_list, filter_contact_list = filter_file_list(all_obj_list, filter_flag, posa_dir=posa_dir)
    filter_img_list = [img_fn_list[tmp_idx] for tmp_idx in range(len(img_list)) if filter_flag[tmp_idx]]
    with open(os.path.join(tmp_save_dir, f'filter_input_list.pickle'), 'wb') as fout:
        pickle.dump( \
            {'filter_obj_list': filter_obj_list, \
            'filter_contact_list': filter_contact_list, \
            'filter_img_list': filter_img_list}, \
        fout)
    return vertices_np_scene, body2scene_conf, filter_flag, \
        ori_body2scene_conf, filter_obj_list, filter_contact_list, filter_img_list
    
def create_template_dir(random_sample_order, save_dir, original_batch_size):
    if random_sample_order == -1:
        template_save_dir = os.path.join(save_dir+ '/../../template', f'{original_batch_size}')
    else:
        template_save_dir = os.path.join(save_dir+ '/../../template', f'{original_batch_size}random{random_sample_order}')
    logger.info(f'template save dir: {template_save_dir}')

    tmp_save_dir = os.path.join(template_save_dir, f'preload_img')
    os.makedirs(tmp_save_dir, exist_ok=True)
    return tmp_save_dir, template_save_dir

def get_tb_logger(save_dir, debug=False):
    if not debug:
        tb_logger = None,
    else:
        from tensorboardX import SummaryWriter
        os.system(f'rm {save_dir}/events.out.tfevents*')
        tb_logger = SummaryWriter(save_dir)
    return tb_logger

# use a static scene det or a moving scene det;
def get_scene_det_path(img_path, img_dir, scene_result_path, scene_result_dir, idx):
    if img_path is None and img_dir is not None: # run single image.
        all_list = sorted(glob.glob(os.path.join(img_dir, '*.png')))
        img_path = all_list[idx]
    
    if scene_result_path is not None:
        det_fn = os.path.join(scene_result_path, f'detections.json')
    else:
        det_fn = os.path.join(scene_result_dir, f'frame{idx}/detections.json')
    
    return img_path, det_fn
        
def get_img_list(opt):
    img_list = opt.pop('img_list', [1])
    random_sample_order = opt.random_sample_order
    all_frame = opt.all_frame
    if -1 in img_list:
        if random_sample_order != -1:
            img_list = [one for one in range(1, all_frame+1)]
            opt.update({'batch_size': all_frame})
            
        else:
            img_list = [one for one in range(1, opt.batch_size+1)]
        opt['img_list'] = img_list
        print('reformulate img_list', img_list)
    else:
        opt['img_list'] = img_list
    return img_list, opt

def get_verts_from_smplifx(st_2_body_model):
    vertices_list = []
    for idx in tqdm(range(len(st_2_body_model)), desc="st2 save body"):
        body_model = st_2_body_model[idx]
        vertices_list.append(body_model.vertices)
        vertices_np = np.stack(vertices_list)
    return vertices_np

def set_optimized_scene_parameters(model, lr_idx, update_gp=False):
    if not update_gp:
        lr_idx += 1
        
    if lr_idx == 0:
        # optimize ground plane and pose
        pass
    elif lr_idx >= 1:
        # fixed gp and camera pose, update objects
        model.set_static_scene(['ground_plane', 'rotate_cam_roll', 'rotate_cam_pitch'])

def render_model_to_imgs(model, image, scene_viz, sub_fn, scanned_scene=None, debug=False):

    if debug:
        result = {}
    # render to image
    if image.max() > 1:
        image = image / 255.0
    rend, mask = model.render()
    h, w, c = image.shape
    L = max(h, w)
    new_image = np.pad(image.copy(), ((0, L - h), (0, L - w), (0, 0)), mode='constant')
    new_image[mask] = rend[mask]
    tmp = new_image.copy()
    new_image = (new_image[:h, :w] * 255).astype(np.uint8)
    
    # add each obj idx, size, position;
    objs_wc = [one.detach().cpu().numpy().mean(axis=1).squeeze() for one in model.get_verts_object_parallel_ground(True)[1]]

    new_image_ = Image.fromarray(new_image, 'RGB')
    new_image_.save(os.path.join(save_dir, f'{sub_fn}.png'))
    logger.info(f'save to {save_dir}')

    if scene_viz:
        plt.imshow(new_image)
        plt.show(block=False)
        plt.pause(0.5)
        plt.close()
    
    if debug:
        result[f'{sub_fn}'] = tmp
    
    if scanned_scene is not None:
        # render_with_scanned scene
        if image.max() > 1:
            image = image / 255.0
        rend, mask = model.render_with_scene(scanned_scene)
        h, w, c = image.shape
        L = max(h, w)
        new_image = np.pad(image.copy(), ((0, L - h), (0, L - w), (0, 0)), mode='constant')
        new_image[mask] = rend[mask]
        tmp = new_image.copy()
        new_image = (new_image[:h, :w] * 255).astype(np.uint8)
        new_image_ = Image.fromarray(new_image, 'RGB')
        new_image_.save(os.path.join(save_dir, f'{sub_fn}_scanned_scene.png'))
        logger.info(f'save to {save_dir}')

        if image.max() > 1:
            image = image / 255.0
        rend = model.top_render_with_scene(scanned_scene)
        new_image = rend
        new_image = (new_image * 255).astype(np.uint8)
        new_image_ = Image.fromarray(new_image, 'RGB')
        new_image_.save(os.path.join(save_dir, f'{sub_fn}_scanned_scene_top.png'))
        
    # top_render
    if image.max() > 1:
        image = image / 255.0
    rend = model.top_render()
    new_image = rend
    new_image = (new_image * 255).astype(np.uint8)
    new_image_ = Image.fromarray(new_image, 'RGB')
    new_image_.save(os.path.join(save_dir, f'{sub_fn}_top.png'))

    if debug:
        result[f'{sub_fn}_top'] = rend

    if scene_viz:
        plt.imshow(new_image)
        plt.show(block=False)
        plt.pause(0.5)
        plt.close()

    import math
    if image.max() > 1:
        image = image / 255.0
    rend = model.side_render(0, 0)
    new_image = rend
    new_image = (new_image * 255).astype(np.uint8)
    new_image_ = Image.fromarray(new_image, 'RGB')
    new_image_.save(os.path.join(save_dir, f'{sub_fn}_side.png'))
    
    if debug:
        result[f'{sub_fn}_side'] = rend
        return result
    if scene_viz:
        plt.imshow(new_image)
        plt.show(block=False)
        plt.pause(0.5)
        plt.close()
    
    return None

def save_mesh_models(model, save_dir):
    os.makedirs(save_dir, exist_ok=True)
    _, cam_all_objs = model.get_verts_object(return_all=True)
    _, verts_parallel_ground_list = model.get_verts_object_parallel_ground(return_all=True)
    all_objs_faces = model.faces_object
    idx_for_faces = model.idx_each_object_face
    for idx in range(len(idx_for_faces)):
        verts = cam_all_objs[idx].detach().cpu().numpy().squeeze()
        if idx == 0:
            start = 0
            end = idx_for_faces[idx]
            faces = all_objs_faces[0, start:end]
        else:
            start = idx_for_faces[idx-1] 
            end = idx_for_faces[idx]
            faces = all_objs_faces[0, start:end] - model.idx_each_object[idx-1]
        faces = faces.squeeze().detach().cpu().numpy()
        mesh_model = trimesh.Trimesh(verts, faces, process=False)
        mesh_model.export(os.path.join(save_dir, f'f{idx:03d}.obj'))
        
        # save world CS objects.
        verts_world = verts_parallel_ground_list[idx].detach().cpu().numpy().squeeze()
        mesh_model_world = trimesh.Trimesh(verts_world, faces, process=False)
        mesh_model_world.export(os.path.join(save_dir, f'f{idx:03d}_world.obj'))
    
    gp_model, gp_model_world = model.get_checkerboard_ground_np()
    gp_model.export(os.path.join(save_dir, f'gp_mesh.obj'))
    gp_model_world.export(os.path.join(save_dir, f'gp_mesh_world.obj'))

def save_scene_model(model, save_dir, cam_inc, tmp_name):
    # save camera
    save_camera(model, save_dir, f'{tmp_name}', cam_inc)
    # output mesh in camera CS
    save_mesh_models(model, os.path.join(save_dir, tmp_name))
    # save parameters
    # warning: merge rotate_cam_pitch/roll into K_extrin
    
    model.K_extrin.data.copy_(model.get_cam_extrin().detach())
    model.rotate_cam_pitch.data.copy_(torch.zeros(1, 1))
    model.rotate_cam_roll.data.copy_(torch.zeros(1, 1))
    torch.save(model.state_dict(), os.path.join(save_dir, f'{tmp_name}.pth'))

def topview_imgs_pyrender(model, image):
    result = {}
    # top_render
    if image.max() > 1:
        image = image / 255.0
    rend = model.top_render_with_scene_pyrender()
    new_image = rend
    
    new_image = (new_image * 255).astype(np.uint8)
    new_image_ = Image.fromarray(new_image, 'RGB')
    result[f'top_view'] = rend
    return result

def write_image_pil(img, save_path, text=None, postion=None):
    # input: np.array;
    # save_path
    if text is not None and position is not None:
        cv2.putText(img, text, postion, cv2.FONT_HERSHEY_COMPLEX,2,(0,255,0),2)
    img = Image.fromarray(img, 'RGB')
    save_dir = os.path.dirname(save_path)
    os.makedirs(save_dir, exist_ok=True)
    # cv2.imwrite(save_path, img)
    img.save(save_path) #scene_human_init
        
def render_model_to_imgs_pyrender(model, image, scene_viz, sub_fn, save_dir=None, scanned_scene=None, debug=False, obj_idx=-1):
    ALPHA = 0.9
    os.makedirs(save_dir, exist_ok=True)
    if debug:
        result = {}
    if scanned_scene is not None and RENDER_SCAN:
        # render_with_scanned scene
        contacted_imgs = []

        if image.max() > 1:
            image = image / 255.0
        
        rend, mask = model.render_with_scene_pyrender(scanned_scene, obj_idx=obj_idx)
        new_image = (rend * mask +
                      (1 - mask) * image)
        new_image = new_image * ALPHA + (1-ALPHA) * image
        new_image = (new_image * 255).astype(np.uint8)
        
        new_image_ = Image.fromarray(new_image, 'RGB')
        if SAVE_ALL_VIEW_IMG:
            tmp_save_all_dir = os.path.join(save_dir, 'all')
            os.makedirs(tmp_save_all_dir, exist_ok=True)
            new_image_.save(os.path.join(tmp_save_all_dir, f'{sub_fn}_scanned_scene.png')) #scene_human_init
            logger.info(f'save to {tmp_save_all_dir}/{sub_fn}')
        contacted_imgs.append(new_image)

        if image.max() > 1:
            image = image / 255.0
        
        # rend = model.top_render_with_scene(scanned_scene)
        rend, mask = model.top_render_with_scene_pyrender(scanned_scene, obj_idx=obj_idx)
        new_image = rend
        new_image = (new_image * 255).astype(np.uint8)
        new_image_ = Image.fromarray(new_image, 'RGB')
        if SAVE_ALL_VIEW_IMG:
            new_image_.save(os.path.join(save_dir, f'{sub_fn}_scanned_scene_top.png'))
        contacted_imgs.append(new_image)
          
        contacted_imgs = Image.fromarray(np.vstack(contacted_imgs), 'RGB')
        if SAVE_SCENE_RENDER_IMG:
            os.makedirs(os.path.join(save_dir, 'scanned'), exist_ok=True)
            contacted_imgs.save(os.path.join(save_dir, 'scanned', f'{sub_fn}_scanned_all.png'))

        if debug:
            result[f'{sub_fn}_scanned'] = np.array(contacted_imgs).astype(np.float) / 255 

    contacted_imgs = []    
    
    # render to image
    ALPHA=0.5
    if image.max() > 1:
        image = image / 255.0
    # mask is always 0, 1
    rend, mask = model.render_with_scene_pyrender(obj_idx=obj_idx)
    new_image = (rend * mask +
                      (1 - mask) * image)
    new_image = new_image * ALPHA + (1-ALPHA) * image
    tmp = new_image
    new_image = (new_image * 255).astype(np.uint8)
    contacted_imgs.append(new_image)
    objs_wc = [one.detach().cpu().numpy().mean(axis=1).squeeze() for one in model.get_verts_object_parallel_ground(True)[1]]

    if DEBUG_OBJ_SIZE:
        for i, one in enumerate(model.size_cls):
            size_info = model.get_size_of_each_objects(idx=i)
            bbox = model.det_results[i, :-1]
            cv2.putText(new_image, size_info, (50, 0+20*i+40), cv2.FONT_HERSHEY_COMPLEX,0.5,(255,0,0),1)
            pos_info = f"{i}:{[float('{:.2f}'.format(obj)) for obj in objs_wc[i].tolist()]}"
            cv2.putText(new_image, pos_info, (bbox[0]-50, bbox[1]+30*i+10), cv2.FONT_HERSHEY_COMPLEX,0.5,(255,0,0),1)

        body_verts = model.verts_person_og.detach().cpu().numpy()
        if model.verts_person_og.shape[0] != 0:
            posed_body_height = body_verts[:, 1].max() - body_verts[:, 1].min()
            position_body = body_verts.mean(0)
            cv2.putText(new_image, f"height: {posed_body_height}, pos: {[float('{:.2f}'.format(obj)) for obj in position_body]}", \
                (10, 10), cv2.FONT_HERSHEY_COMPLEX, 0.5,(255,0,0),1)
        
    new_image_ = Image.fromarray(new_image, 'RGB')
    os.makedirs(save_dir, exist_ok=True)
    tmp_save_align_dir = os.path.join(save_dir, 'align')
    os.makedirs(tmp_save_align_dir, exist_ok=True)

    new_image_.save(os.path.join(tmp_save_align_dir, f'{sub_fn}.png'))
    logger.info(f'save to {tmp_save_align_dir}')
    if SAVE_APARENT_IMG:
        apparent_img = np.concatenate([rend, mask], -1)
        apparent_img = (apparent_img * 255).astype(np.uint8)
        apparent_img_ = Image.fromarray(apparent_img)
        
        tmp_save_align_alpha_dir = os.path.join(save_dir, 'align_alpha')
        os.makedirs(tmp_save_align_alpha_dir, exist_ok=True)
    
        apparent_img_.save(os.path.join(tmp_save_align_alpha_dir, f'{sub_fn}_alpha.png'))

    if scene_viz:
        plt.imshow(new_image)
        plt.show(block=False)
        plt.pause(0.5)
        plt.close()
    
    # top_render
    if image.max() > 1:
        image = image / 255.0
    rend, mask = model.top_render_with_scene_pyrender(obj_idx=obj_idx)
    new_image = rend
    
    new_image = (new_image * 255).astype(np.uint8)
    new_image_ = Image.fromarray(new_image, 'RGB')
    if SAVE_ALL_VIEW_IMG:
        tmp_save_top_dir = os.path.join(save_dir, 'top')
        os.makedirs(tmp_save_top_dir, exist_ok=True)
        new_image_.save(os.path.join(tmp_save_top_dir, f'{sub_fn}_top.png'))

    if SAVE_APARENT_IMG:
        apparent_img = np.concatenate([rend, mask], -1)
        apparent_img = (apparent_img * 255).astype(np.uint8)
        apparent_img_ = Image.fromarray(apparent_img)
        tmp_save_top_alpha_dir = os.path.join(save_dir, 'top_alpha')
        os.makedirs(tmp_save_top_alpha_dir, exist_ok=True)
        apparent_img_.save(os.path.join(tmp_save_top_alpha_dir, f'{sub_fn}_top_alpha.png'))

    contacted_imgs.append(new_image)

    if scene_viz:
        plt.imshow(new_image)
        plt.show(block=False)
        plt.pause(0.5)
        plt.close()

    # side_render
    import math
    if image.max() > 1:
        image = image / 255.0

    rend, mask = model.side_render_with_scene_pyrender()
    new_image = rend
    new_image = (new_image * 255).astype(np.uint8)
    new_image_ = Image.fromarray(new_image, 'RGB')

    if SAVE_ALL_VIEW_IMG:
        tmp_save_side_dir = os.path.join(save_dir, 'side')
        os.makedirs(tmp_save_side_dir, exist_ok=True)
        new_image_.save(os.path.join(tmp_save_side_dir, f'{sub_fn}_side.png'))
    
    if SAVE_APARENT_IMG:    
        apparent_img = np.concatenate([rend, mask], -1)
        apparent_img = (apparent_img * 255).astype(np.uint8)
        apparent_img_ = Image.fromarray(apparent_img)
        tmp_save_side_alpha_dir = os.path.join(save_dir, 'side_alpha')
        os.makedirs(tmp_save_side_alpha_dir, exist_ok=True)
        apparent_img_.save(os.path.join(tmp_save_side_alpha_dir, f'{sub_fn}_side_alpha.png'))

    contacted_imgs.append(new_image)

    # right side
    rend, mask = model.right_side_render_with_scene_pyrender()
    new_image = rend
    new_image = (new_image * 255).astype(np.uint8)
    new_image_ = Image.fromarray(new_image, 'RGB')
    if SAVE_ALL_VIEW_IMG:
        tmp_save_right_side_dir = os.path.join(save_dir, 'right_side')
        os.makedirs(tmp_save_right_side_dir, exist_ok=True)
        new_image_.save(os.path.join(tmp_save_right_side_dir, f'{sub_fn}_right_side.png'))
    
    if SAVE_APARENT_IMG:
        apparent_img = np.concatenate([rend, mask], -1)
        apparent_img = (apparent_img * 255).astype(np.uint8)
        apparent_img_ = Image.fromarray(apparent_img)
        tmp_save_right_side_alpha_dir = os.path.join(save_dir, 'right_side_alpha')
        os.makedirs(tmp_save_right_side_alpha_dir, exist_ok=True)
        apparent_img_.save(os.path.join(tmp_save_right_side_alpha_dir, f'{sub_fn}_right_side_alpha.png'))

    contacted_imgs.append(new_image)

    contacted_imgs = Image.fromarray(np.vstack(contacted_imgs), 'RGB')
    if SAVE_SCENE_RENDER_IMG:
        os.makedirs(os.path.join(save_dir, 'estimated'), exist_ok=True)
        contacted_imgs.save(os.path.join(save_dir, 'estimated', f'{sub_fn}.png'))
    
    if debug:
        result[f'{sub_fn}_estimated'] = np.array(contacted_imgs).astype(np.float) / 255 
        return result

    if scene_viz:
        plt.imshow(new_image)
        plt.show(block=False)
        plt.pause(0.5)
        plt.close()
    
    return None

def output_render_result(filter_img_list, vertices_np_scene, model, save_dir, tmp_name, \
            device, scanned_scene=None, filter_list=None, tb_logger=None, tb_debug=False, \
            save_video=True):
    for idx in tqdm(range(vertices_np_scene.shape[0])):
        vertices = np.asarray(vertices_np_scene[idx], dtype=np.float32)
        r_vertices = torch.from_numpy(vertices).type_as(model.verts_person_og)
        
        # visualize
        model.verts_person_og = r_vertices
        one_img = get_image(filter_img_list[idx])
        if one_img.max() <= 1.01:
            one_img = (one_img * 255).astype(np.int)
        
        if filter_list is not None:
            save_name = f'{os.path.basename(filter_img_list[idx])[:-4]}_Filter{filter_list[idx]}'
        else:
            save_name = f'{os.path.basename(filter_img_list[idx])[:-4]}'
        
        tmp_result = render_model_to_imgs_pyrender(model, one_img, False, save_name, \
                            save_dir=os.path.join(save_dir, f'{tmp_name}'), debug=tb_debug, scanned_scene=scanned_scene)
        if tb_debug and tb_logger is not None:
            tb_tmp_name = os.path.basename(save_dir)
            save_images(tb_logger, f'body_{idx}_{tb_tmp_name}', tmp_result, idx)

        if RENDER_DEBUG:
            break;

    # ffmpeg them into a video
    if save_video:
        logger.info(f'save video to {save_dir}/{tmp_name}/estimated_out.mp4')
        os.system(f"/usr/bin/ffmpeg -pattern_type glob -i '{save_dir}/{tmp_name}/estimated/*.png'  -c:v libx264 -vf fps=25 -pix_fmt yuv420p {save_dir}/{tmp_name}/estimated_out.mp4 -y")
        if scanned_scene is not None:
            logger.info(f'save video to {save_dir}/{tmp_name}/scanned_out.mp4')
            os.system(f"/usr/bin/ffmpeg -pattern_type glob -i '{save_dir}/{tmp_name}/scanned/*.png'   -c:v libx264 -vf fps=25 -pix_fmt yuv420p  {save_dir}/{tmp_name}/scanned_out.mp4 -y")
                            
def get_scalenet_parameters(cam_fn):
    result = joblib.load(cam_fn)
    return [result['roll'], result['pitch']] 

def save_camera(model, save_dir, save_fn, ori_cam):
    cam_extrin = model.get_cam_extrin().detach().cpu().numpy()
    from scipy.spatial.transform import Rotation as R
    euler_angle = R.from_matrix(cam_extrin).as_euler('zyx')
    with open(os.path.join(save_dir, f'{save_fn}.json'), 'w') as cam_fout:
        json.dump(cam_extrin.tolist(), cam_fout)
    with open(os.path.join(save_dir, f'{save_fn}_euler.json'), 'w') as cam_fout:
        json.dump(euler_angle.tolist(), cam_fout)
    print('save to ', os.path.join(save_dir, f'{save_fn}_001.xml'))
    write_opencv_matrix(os.path.join(save_dir, f'{save_fn}_001.xml'), ori_cam, cam_extrin[0])


### TODO: finalize with constants.pys
def get_setting_scene(stage3_kind_flag, opt=None):
    ################################
    ### kind flag: 1 run single/multiple image as test without filter
    ###            2 run whole video
    ################################
    if stage3_kind_flag == 0: # ! for multi-imgs, filter=False
        
        filter_open = False
        video_process = False
        st3_lr_list = [2e-3, 2e-3]
        st3_num_iterations_list = [1000, 1500]
        # st3_pid_list = [20119+opt.process_id]
        st3_pid_list = [10127, 17000]
        # st3_pid_list = [90354+opt.process_id]
        segments_list = [1] # each element defines how many segments for each video.
        fps_list = [6]

    elif stage3_kind_flag == 1:
        filter_open = True
        video_process = True
        st3_lr_list = [2e-3, 2e-3]
        st3_num_iterations_list = [1000, 2500] 
        st3_pid_list = [10127, 17000]
        segments_list = [1] # each element defines how many segments for each video.
        fps_list = [6]
    else:
        logger.info('wrong stage3_kind_flag')
        assert False

    return {"filter_open": filter_open,
        "video_process": video_process,
        "st3_lr_list": st3_lr_list,
        "st3_num_iterations_list": st3_num_iterations_list,
        "st3_pid_list": st3_pid_list, 
        "segments_list": segments_list,
        "fps_list": fps_list,
        }

def load_scene_init(scene_init_model, model, load_all_scene, update_gp_camera, noise_kind=-1, noise_value=0.0):
    logger.info('load estimated camera and ground plane !!!')
    logger.info(f'from {scene_init_model}')
    assert os.path.exists(scene_init_model)
    if scene_init_model is not None and os.path.exists(scene_init_model):
        logger.info('load model:',scene_init_model)
        if not load_all_scene: # only update camera and gp
            update_list = ['ground_plane', 'rotate_cam_roll', 'rotate_cam_pitch', 'K_extrin'] 
            # ignore_list=[]
            #['translations_object', 'rotations_object', 'int_scales_object']
            model.load_scene_init(torch.load(scene_init_model), update_list=update_list)
        else:
            model.load_scene_init(torch.load(scene_init_model), ignore_list=['det_results', 'det_score'])
        
        # add noise on input
        if noise_kind != -1:
            logger.info(f'Add noise to input, kind {noise_kind}')
            model.add_noise(noise_kind, noise_value)
    else:
        logger.info('not exist model:', scene_init_model)
    
    model.input_body_flag=False
    model.input_body_contact_flag=False
    model.depth_template_flag = False

    if not update_gp_camera:  # ! update camera, this is real useful for update camera.
        logger.info('camera is static !!!!')
        model.set_static_scene(['ground_plane', 'rotate_cam_roll', 'rotate_cam_pitch'])

def load_feet_gp_contact(body_segments_dir):
    ground_contact_vertices_ids = []
    for part in [ 'L_feet_front', 'L_feet_back', 'R_feet_front', 'R_feet_back']:
        with open(os.path.join(body_segments_dir, part + '.json'), 'r') as f:
            data = json.load(f)
            ground_contact_vertices_ids.append(list(set(data["verts_ind"])))
    ground_contact_vertices_ids = np.stack(ground_contact_vertices_ids)
    return ground_contact_vertices_ids

def save_filter_render_body_results(all_obj_list, filter_flag, save_dir, render_dir):
    # save without scan render results.
    filter_obj_list = [all_obj_list[tmp_idx] for tmp_idx in range(len(filter_flag)) if filter_flag[tmp_idx]]
    os.makedirs(os.path.join(save_dir, 'body_filter'), exist_ok=True)
    for one in filter_obj_list:
        base_fn = int(os.path.basename(one).split('.')[0])
        src_fn = os.path.join(render_dir, f'{base_fn:06d}.png')
        dst_fn = os.path.join(save_dir, 'body_filter',f'{base_fn:06d}.png')
        os.system(f'ln -s {src_fn} {dst_fn}')


def filter_file_list(all_obj_list, filter_flag, posa_dir=None):
    
    filter_obj_list = [all_obj_list[tmp_idx] for tmp_idx in range(len(filter_flag)) if filter_flag[tmp_idx]]
    filter_contact_list = []
    if posa_dir is not None:
        for obj_fn in filter_obj_list:
            basename = os.path.basename(obj_fn)
            idx = int(basename.split('.')[0])
            filter_contact_list.append(os.path.join(posa_dir, f'{idx-1:06d}_sample_00.npy'))
    return filter_obj_list, filter_contact_list

def filter_human_poses(filter_open, video_process, st_2_fit_body_use_dict, vertices_np,
            per_segment_fps, start_frame, end_frame, new_pre_smplx_model, save_dir, 
            s_i, num_segments,
            img_list, tb_debug, tb_logger,
            use_filter_conf=False, 
            model=None, scanned_scene=None, op_filter=None):
                
    if filter_open: ## ! warning: use filter_flag to select frames in stage3.
        if st_2_fit_body_use_dict is None:
            keypoints_2d_conf = st_2_fit_body_use_dict['keypoints'][..., -1].cpu().numpy()
        else:
            keypoints_2d_conf = np.ones((vertices_np.shape[0], 25))
        from mover.utils.filter_body_pose import filter_body_for_imgs, filter_body_for_video
        if not video_process:
            vertices_np_scene, body2scene_conf, filter_flag, ori_body2scene_conf = filter_body_for_imgs(vertices_np, keypoints_2d_conf, thre=0.70)#, top=20) # 0.25
        else:
            if model is not None:
                ground_plane = model.ground_plane.detach().cpu().numpy()
            else:
                ground_plane = 10.0
            if 'keypoints_3d' not in new_pre_smplx_model.keys():
                kpts_3d_idx = KPTS_OPENPOSE_FOR_SMPLX
                tmp_keypoint_3d = vertices_np[:, kpts_3d_idx, :]
            else:
                tmp_keypoint_3d = new_pre_smplx_model['keypoints_3d']
            
            vertices_np_scene, body2scene_conf, filter_flag, ori_body2scene_conf = filter_body_for_video(vertices_np, keypoints_2d_conf, thre=0.50, fps=per_segment_fps, \
                                start_fr=start_frame, end_fr=end_frame, \
                                ori_joints_3d=tmp_keypoint_3d, \
                                debug=True, save_dir=save_dir, segment_idx=s_i, segment_num=num_segments, whole_video=True, ground_plane=ground_plane, op_filter=op_filter)
    else:
        vertices_np_scene = vertices_np
        body2scene_conf = np.ones(vertices_np.shape[0]).astype(vertices_np.dtype)
        filter_flag = np.ones(vertices_np.shape[0]) == 1.0
        ori_body2scene_conf = body2scene_conf.astype(vertices_np.dtype)
        
        if op_filter is not None:
            vertices_np_scene = vertices_np_scene[op_filter.nonzero()[:,0]]
            body2scene_conf = body2scene_conf[op_filter.nonzero()[:,0]]
            filter_flag[(op_filter==0).nonzero()[:,0]] = 0
            ori_body2scene_conf = ori_body2scene_conf[op_filter.nonzero()[:,0]]

    if not use_filter_conf: 
        body2scene_conf = np.ones(vertices_np_scene.shape[0]).astype(vertices_np_scene.dtype)

    message_flag = []
    tmp_img_list = np.array(img_list)[filter_flag]
    for c_idx in range(body2scene_conf.shape[0]):
        tmp_info = f'{c_idx}_{tmp_img_list[c_idx]}:{body2scene_conf[c_idx]}_{ori_body2scene_conf[c_idx]}\n'
        message_flag.append(tmp_info)
        
    with open(os.path.join(save_dir, 'st3_filter_conf_ori_conf.txt'), 'w') as fout:
        fout.writelines(''.join(message_flag))

    if tb_debug:
        message_flag = ''
        tmp_img_list = np.array(img_list)[filter_flag]
        for f_flag in range(tmp_img_list.shape[0]):
            message_flag += f'{f_flag}: {tmp_img_list[f_flag]} \n'
        tb_logger.add_text('filter flag', message_flag, 0)
        logger.info(f'filter num: {filter_flag.sum()}')         

    return vertices_np_scene, body2scene_conf, filter_flag, ori_body2scene_conf