# defined using numpy
import numpy as np
import math
import os 
from scipy.ndimage.morphology import binary_dilation,binary_erosion

def filter_body_for_video(ori_verts_np, ori_op_conf, thre=0.25, fps=50, start_fr=0, end_fr=-1, \
                ori_joints_3d=None, debug=True, save_dir=None, segment_idx=0, segment_num=1, \
                whole_video=False, ground_plane=None, op_filter=None):

    # split segment for a video
    # TODO: use start_fr and end_fr
    all_frames = ori_verts_np.shape[0]
    interval = segment_num * fps
    tmp_num = math.ceil(all_frames / interval)
    first_select_idx = [min(i*interval+segment_idx*fps, all_frames) for i in range(tmp_num)]

    verts_np = ori_verts_np
    op_conf = ori_op_conf
    joints_3d = ori_joints_3d

    # calculate the speed, acceleration on 3D joints
    tmp_list = [2,5, 9, 12, 10, 13, 11, 14] # should + Hip + Knee + Ankle: Torse Joint
    tmp_joints = joints_3d[:, tmp_list]
    speed = (tmp_joints[1:, :] - tmp_joints[:-1, :]).mean(1)
    diff_transl = tmp_joints[1:, :] - tmp_joints[:-1, :]
    accelaration = diff_transl[1:, :] - diff_transl[:-1, :]
    accelaration = accelaration.mean(1) # three dim: x,y,z
    
    # translation error: pelvis 
    pelvis_list = [8]
    pel_joints = joints_3d[:, pelvis_list]
    pel_s = ( pel_joints[1:, :] - pel_joints[:-1, :]).mean(1)
    diff_s = ( pel_joints[1:, :] - pel_joints[:-1, :])
    pel_a = (diff_s[1:, :] - diff_s[:-1, :]).mean(1)
    # pose prior: relative to pelvis
    relative_joints = tmp_joints-pel_joints
    relative_speed = (relative_joints[1:, :] - relative_joints[:-1, :]).mean(1)
    relative_diff_transl = relative_joints[1:, :] - relative_joints[:-1, :]
    relative_accelaration = (relative_diff_transl[1:, :] - relative_diff_transl[:-1, :]).mean(1) # three dim: x,y,z
    
    # useful confidence
    use_list = [8, 9, 12, 10, 13, 11, 14] #[1, 2, 5, 3, 6, 4, 7] pelvis + RHip + Lhip + Knee + Ankle
    min_conf = op_conf[:, use_list].min(-1)
    use_list_torso = [8, 9, 12, 10, 13, 11, 14, 2, 5, 3, 6, 4, 7] # torso + arm
    avg_conf = op_conf[:, use_list_torso].mean(-1)
    
    # score
    yspeed = np.linalg.norm(speed, ord=2, axis=-1)
    yaccelaration = np.linalg.norm(accelaration, ord=2, axis=-1)
    
    # transl
    pel_yspeed = np.linalg.norm(pel_s, ord=2, axis=-1)
    pel_yaccelaration = np.linalg.norm(pel_a, ord=2, axis=-1)
    
    # relative
    relative_yspeed = np.linalg.norm(relative_speed, ord=2, axis=-1)
    relative_yaccelaration = np.linalg.norm(relative_accelaration, ord=2, axis=-1)

    all_batch = accelaration.shape[0]    

    # accelaraton
    pel_yaccelaraton_reverse = np.clip(0.025-pel_yaccelaration, 0, 0.025) / 0.025
    relative_yaccelaration_reverse = np.clip(0.025-relative_yaccelaration, 0, 0.025) / 0.025

    tmp_min_conf = min_conf[:all_batch]
    tmp_avg_conf = avg_conf[:all_batch]
    
    # Filter Pose Kind 1: select the best one in a small window frames
    KIND=2
    if KIND == 2:
        # accelaration = t2+t0-2*t1
        # * used without motion prior smooth loss: threshold = 0.06
        pel_thershold = 0.18
        acc_thershold = 0.09 

        trans_filter = pel_yaccelaration < pel_thershold
        local_filter = relative_yaccelaration < acc_thershold
        # import pdb;pdb.set_trace()
        # dilation. TODO: image processs.
        trans_filter = binary_erosion(trans_filter)
        local_filter = binary_erosion(local_filter)

        all_filter = trans_filter & local_filter
        # ! add ground plane filter.
        if ground_plane is not None:
            # import pdb;pdb.set_trace()
            ground_plane_np = ground_plane
            gp_filter = (np.max(verts_np[:, :, 1], -1) - ground_plane_np) < 0.1
            all_filter = all_filter & gp_filter[:-2]
            
        all_conf_np = pel_yaccelaration * relative_yaccelaration
        all_conf_np[all_filter] = 100
        new_filter_idx = []
        for i, img_idx in enumerate(first_select_idx):
            # import pdb;pdb.set_trace()
            # all_conf = all_conf_np[img_idx:img_idx+fps][all_filter[img_idx:img_idx+fps]] #img_idx:min(img_idx+fps, all_filter.shape[0])
            all_conf = all_conf_np[img_idx:img_idx+fps] #img_idx:min(img_idx+fps, all_filter.shape[0])
            if all_conf.shape[0] == 0:
                continue
            arg_max_i = np.argmin(all_conf) + img_idx
            if all_filter[arg_max_i]:
                if op_filter is not None and not op_filter[arg_max_i]: # op confidence=0
                    continue
                new_filter_idx.append(arg_max_i)
        
        local_new_filter_flag = np.zeros(verts_np.shape[0], dtype=np.bool)
        indicator_5 = np.zeros(verts_np.shape[0], dtype=np.bool)
        indicator_5[np.arange(0, verts_np.shape[0], 5)] = True
        local_new_filter_flag[new_filter_idx] = True
        verts_np = verts_np[local_new_filter_flag]
        ori_use_conf = np.ones(verts_np.shape[0])
        use_conf = np.ones(verts_np.shape[0])
    if debug: 
        import matplotlib.pyplot as plt
        show_w = 100
        tmp_n = math.ceil(accelaration.shape[0]/show_w)

        for i in range(tmp_n):
            start = i * show_w # start from 1 
            end = min((i+1)*show_w, accelaration.shape[0])
            xaxis = np.arange(accelaration.shape[0])[start:end]

            yspeed_tmp = np.linalg.norm(speed[xaxis], ord=2, axis=-1)
            yaccelaration_tmp = np.linalg.norm(accelaration[xaxis], ord=2, axis=-1)
            
            # transl
            pel_yspeed_tmp = np.linalg.norm(pel_s[xaxis], ord=2, axis=-1)
            pel_yaccelaration_tmp = np.linalg.norm(pel_a[xaxis], ord=2, axis=-1)
            
            # relative
            relative_yspeed_tmp = np.linalg.norm(relative_speed[xaxis], ord=2, axis=-1)
            relative_yaccelaration_tmp = np.linalg.norm(relative_accelaration[xaxis], ord=2, axis=-1)

            # filter results
            local_new_filter_flag_tmp = local_new_filter_flag[xaxis] * 0.5

            # indicator 5 axis
            indicator_5_tmp = indicator_5[xaxis] * 0.8
            
            # fig 1
            fig = plt.figure(figsize=(12, 12), dpi=80,)
            ax1 = fig.add_subplot(311)
            ax1.plot(xaxis, local_new_filter_flag_tmp, 'r*-', label='filter results')
            tmp_xaxis = xaxis[0:-1:5]
            
            ax1.set_xlabel(f'filter with pel_acce < {pel_thershold} & local_acce < {acc_thershold}, filter results: {local_new_filter_flag.sum()}/{local_new_filter_flag.shape[0]}')
            ax1.set_xticks(tmp_xaxis)
            ax1.set_ylabel('filter flag')
            ax1.legend()

            # fig 2
            ax1 = fig.add_subplot(312)
            ax1.plot(xaxis, pel_yspeed_tmp, 'r*-', label='pel_speed')
            ax1.plot(xaxis, pel_yaccelaration_tmp, 'b^-', label='pel_accelaration')
            ax1.set_xticks(tmp_xaxis)
            ax2 = ax1.twinx()
            yconf = min_conf[xaxis]
            ax2.plot(xaxis, yconf, 'g*-', label='conf')
            ax2.tick_params(axis='y')
            
            ax1.set_xlabel('frames along the video for Pelvis Joint')
            ax1.set_ylabel('speed & accelaration')
            ax1.legend()
            ax2.set_ylabel('conf')
            ax2.legend(loc=0)

            # fig 3
            ax1 = fig.add_subplot(313)
            ax1.plot(xaxis, relative_yspeed_tmp, 'r*-', label='relative_speed')
            ax1.plot(xaxis, relative_yaccelaration_tmp, 'b^-', label='relative_accelaration')
            ax1.set_xticks(tmp_xaxis)

            ax2 = ax1.twinx()
            yconf = min_conf[xaxis]
            ax2.plot(xaxis, yconf, 'g*-', label='conf')
            ax2.tick_params(axis='y')
            ax1.set_xlabel('frames along the video for relative pose transformation')
            ax1.set_ylabel('speed & accelaration')
            ax1.legend()
            ax2.set_ylabel('conf')
            ax2.legend(loc=0)

            print('save to ', os.path.join(save_dir, f'conf_joints_speed_accelaration_{i}.jpg'))
            fig.savefig(os.path.join(save_dir, f'conf_joints_speed_accelaration_{i}.jpg'),
                format='jpeg',
                dpi=100,
                bbox_inches='tight')

    return verts_np, use_conf, local_new_filter_flag, ori_use_conf
    


def filter_body_for_imgs(verts_np, op_conf, thre=0.25, top=None):
    # input conf: b*118
    use_list = [8, 9, 12, 10, 13, 11, 14] #[1, 2, 5, 3, 6, 4, 7] upper body
    tmp_use_conf = op_conf[:, use_list]

    min_conf = tmp_use_conf.min(-1)
    filter_flag = min_conf > thre
    if top is not None:
        top_min_conf_idx = min_conf.argsort()[-top:]
        tmp_filter = np.zeros(min_conf.shape[0]) != 0.0
        tmp_filter[top_min_conf_idx] = True
        
        filter_flag = np.array([True  if a and b else False for a, b in zip(tmp_filter, filter_flag)])
            
    use_conf = min_conf[filter_flag]    
    verts_np = verts_np[filter_flag]
    ori_use_conf = use_conf.copy()
    use_conf = use_conf / use_conf.sum() * use_conf.shape[0]

    return verts_np, use_conf, filter_flag, ori_use_conf