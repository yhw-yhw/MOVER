import sys
import os
from glob import glob
from tqdm import tqdm
import trimesh
import numpy as np

total_label = [4, 5, 6, 7]
occnet_label = ['02818832', '03001627', '04256520', '04379243']
label_name = ['bed', 'chair', 'sofa', 'table']

def prepare_occupancy_input(input_dir, video_name, input_name='Total3D'):
    save_dir = os.path.join(input_dir, video_name, f'{input_name}_occNet_input')
    all_obj_list = glob(os.path.join(input_dir, video_name, f'{input_name}/*.obj'))
    avalible_list = []
    test_flag = set()
    for one in all_obj_list:
        fn = os.path.basename(one)[:-4]
        idx, kind = int(fn.split('_')[0]), int(fn.split('_')[1])
        if kind in total_label:
            target_dir = os.path.join(save_dir, occnet_label[total_label.index(kind)])
            os.makedirs(target_dir, exist_ok=True)
            os.system(f'cp {one} {target_dir}')
            if kind not in test_flag and os.path.exists(os.path.join(target_dir, 'test.lst')):
                os.remove(os.path.join(target_dir, 'test.lst'))
            if kind not in test_flag:
                with open(os.path.join(target_dir, 'test.lst'), 'a') as fout:
                    fout.write(f'{fn}')
            else:
                with open(os.path.join(target_dir, 'test.lst'), 'a') as fout:
                    fout.write(f'\n{fn}')
            test_flag.add(kind)
            
    # write new yaml
    src_yaml = '/is/cluster/hyi/workspace/Multi-IOI/occupancy_networks/configs/demo_obj.yaml'
    target_yaml = os.path.join(input_dir, video_name, f'{input_name}_occNet_input', 'demo_obj.yaml')
    with open(src_yaml, 'r') as fin:
        all_lines = fin.readlines()
    all_lines[3] = f'  path: {save_dir}\n'
    tmp_save_dir = save_dir.replace(f'{input_name}_occNet_input', f'{input_name}_occNet_results')
    all_lines[8] = f'  out_dir: {tmp_save_dir}\n'
    with open(target_yaml, 'w') as fout:
        for i in all_lines:
            fout.write(i)


def postprocess_occupancy_result(input_dir, video_name, input_name='Total3D'):
    root_dir = os.path.join(input_dir, video_name, f'{input_name}_occNet_results/generation/')
    mesh_input_dir = root_dir + 'meshes'
    save_dir = root_dir + 'all_modify'
    os.makedirs(save_dir, exist_ok=True)
    # from pathlib import Path
    # import pdb;pdb.set_trace()
    # for path in Path(mesh_input_dir).rglob('*.off'):
    #     print(path.name)
    file_list = glob(os.path.join(mesh_input_dir, '**/*.off'))
    # import pdb;pdb.set_trace()

    # save to new Total3D directory.
    target_dir = os.path.join(input_dir, video_name, f'{input_name}_occNet_results/Total3D_input')
    src_dir = os.path.join(input_dir, video_name, f'{input_name}')
    os.makedirs(target_dir, exist_ok=True)
    os.system(f"cp -r {src_dir}/* {target_dir}")

    for file_path in tqdm(file_list):
        
        mesh = trimesh.load(file_path, process=False)
        vert = mesh.vertices * 2
        face = mesh.faces
        new_vert = np.stack([vert[:, 2], vert[:, 1], vert[:, 0]], axis=1)
        save_path = os.path.join(save_dir, os.path.basename(file_path).replace('off', 'obj'))
        new_mesh = trimesh.Trimesh(new_vert, face,
                                process=False)
        new_mesh.export(save_path)

        # ! save to target_dir
        os.system(f'cp {save_path} {target_dir}')

        # TODO: compare P-P distance to filter those pure outlier points.
        # input_p_path = os.path.basename(file_path).replace('mesh.off', 'in.ply')
        # input_ply = trimesh.load(input_p_path, process=False)
        # input_verts = input_ply.vertices
        # filter_mesh = filter(input_verts, new_mesh)
        # save_filter_path = save_path + '_filter.obj'
        # filter_mesh.export(save_filter_path)
        try:
            smpl_list = [1000, 2000, 4000]
            for one in smpl_list:
                wt_smplify_save_fn = save_path + f'_{one}f.obj'
                os.system('../Manifold/build/simplify -i '+ save_path + ' -o ' + wt_smplify_save_fn + f' -f {one}')
        except:
            print('run on workstation.')
    
if __name__ == '__main__':
    input_dir = sys.argv[1]
    flag = sys.argv[2]
    try:
        input_name = sys.argv[3]
    except:
        input_name='Total3D'
    video_name = os.path.basename(input_dir)
    sub_input_dir = os.path.dirname(input_dir)
    
    if flag == 'pre':
        prepare_occupancy_input(sub_input_dir, video_name, input_name)
    elif flag == 'post':
        postprocess_occupancy_result(sub_input_dir, video_name, input_name)
    else:
        assert False
