import trimesh
import numpy as np

def viz_verts(self, verts, save_path, verts_n=None, verts_c=None, camera_mat=None):
    
    assert save_path.split('.')[-1] == 'ply'

    if verts_c is not None:
        out_mesh = trimesh.PointCloud(verts, \
            colors=verts_c, process=False)
        out_mesh.export(save_path) 
    else:
        out_mesh = trimesh.Trimesh(verts, \
            vertex_normals=verts_n, \
            process=False)
        out_mesh.export(save_path,vertex_normal=True) 
    
    if camera_mat is not None:
        camera_verts = np.matmul(camera_mat,verts.T).T
        

        tmp_save_path = save_path[:-4] + '_cameraCS.ply'
        if verts_c is not None:
            out_mesh = trimesh.PointCloud(camera_verts, \
                colors=verts_c, process=False)
            out_mesh.export(tmp_save_path) 
        elif verts_n is not None:
            camera_verts_n = np.matmul(camera_mat, verts_n.T).T
            out_mesh = trimesh.Trimesh(camera_verts, \
                vertex_normals=camera_verts_n, \
                process=False)
            out_mesh.export(tmp_save_path,vertex_normal=True)

    print(f'save to {save_path}')