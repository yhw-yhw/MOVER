from mover.constants import (
    NYU40CLASSES,
)
import numpy as np

# def save_obj(self, fname):
#     with open(fname, "w") as fp:
#         verts_combined = combine_verts(
#             [self.get_verts_object(), self.get_verts_person()]
#         )
#         for v in tqdm.tqdm(verts_combined[0]):
#             fp.write(f"v {v[0]:f} {v[1]:f} {v[2]:f}\n")
#         o = 1
#         for f in tqdm.tqdm(self.faces[0]):
#             fp.write(f"f {f[0] + o:d} {f[1] + o:d} {f[2] + o:d}\n")


def get_size_number_objects(self):
    if self.USE_ONE_DOF_SCALE:
        # import pdb;pdb.set_trace()
        int_scales_object = self.get_scale_object()[:,0:1].expand(self.int_scales_object.shape[0],3)
    else:
        int_scales_object = self.get_scale_object()

    objs_size = self.ori_objs_size * int_scales_object
    return objs_size


# ! Warning: evaluate 3D Scene in camera coordinates system
def get_evaluate_3d_scene(self, gt_bboxes):
    # TODO: add 2D bbox assigned parts
    # depth info (m)
    verts = self.get_verts_object(True)

    objs_center = np.stack([one.detach().cpu().numpy().mean(axis=1).squeeze() for one in verts[1]])
    objs_size = self.get_size_number_objects().detach().cpu().numpy()
    
    # import pdb;pdb.set_trace()
    size_error = np.abs(objs_size - gt_bboxes['coeffs']).mean()
    center_transl_error = np.abs(objs_center - gt_bboxes['centroid']).mean()
    depth_error = np.abs(objs_center - gt_bboxes['centroid'])[:, -1].mean()

    # TODO: add 3D IoU, 2D IoU, Pixel Error
    return {
        'size_error': size_error,
        'center_transl_error': center_transl_error,
        'depth_error': depth_error,
    }


def get_size_of_each_objects(self, idx=None):
    # message = ['*****']
    message = []
    if self.USE_ONE_DOF_SCALE:
        # import pdb;pdb.set_trace()
        int_scales_object = self.get_scale_object()[:,0:1].expand(self.int_scales_object.shape[0],3)
    else:
        int_scales_object = self.get_scale_object()

    objs_size = self.ori_objs_size * int_scales_object
    if idx is None:
        for i, one in enumerate(self.size_cls): # index in NYU40CLASSES
            # gt_size_objs.append(SIZE_FOR_DIFFERENT_CLASS[NYU40CLASSES[one]])
            message.append(f"{i}: {NYU40CLASSES[one]} {int_scales_object.detach().cpu().numpy()[i]} {objs_size[i].detach().cpu().numpy().tolist()}    ") #/}
    else:
        i = idx
        one = self.size_cls[i]
        message.append(f"{i}: {int_scales_object.detach().cpu().numpy()[i]} {[float('{:.2f}'.format(one)) for one in objs_size[i].detach().cpu().numpy().tolist()]}    ")
    return ''.join(message)
    