import os.path as osp

EXTERNAL_DIRECTORY = "./external"

# Configurations for mover
FOCAL_LENGTH = 1.0
IMAGE_SIZE = 640 # 1920 for image fusing.
# IMAGE_SIZE = 1920 
REND_SIZE = 256  # Size of target masks for silhouette loss.
BBOX_EXPANSION_FACTOR = 0.3  # Amount to pad the target masks for silhouette loss.
# SMPL_FACES_PATH = "models/smpl_faces.npy" # MODIFY to SMPLX
SMPL_FACES_PATH = "../data/smpl-x_model/models/smplx/SMPLX_MALE.pkl"

###### debug config
DEBUG=False
DEBUG_LOSS=False # save into tensorboard

DEBUG_LOSS_OUTPUT=False
DEBUG_DEPTH_LOSS=False 
SAVE_DEPTH_MAP=True

DEBUG_DEPTH_PEROBJ_MASK=False
DEBUG_FILTER_POSE=False
DEBUG_CONTACT_LOSS=True
DEBUG_OBJ_SIZE=False


###### vposer version.
USE_PROX_VPOSER=False
USE_2022_VPOSER=True

VIZ_POSA_CONTACT=True
RENDER_DEBUG=False
RENDER_WITH_ALPHA_IMG=True
TB_DEBUG=True
SAVE_BEST_MODEL=False
REMOVE_OBJS=True

# contact loss
ADD_HAND_CONTACT=False
USE_HAND_CONTACT_SPLIT=True

# define loss
LOSS_NORMALIZE_PER_ELEMENT=False

###### output results for paper draft
SAVE_ALL_VIEW_IMG=False # output *_top.png for paper
SAVE_SCENE_RENDER_IMG=True # output estimated/scanned for visualization
SAVE_APARENT_IMG=True

# render scene & body results.
SAVE_ALL_RENDER_RESULT=True
ONLY_SAVE_FILETER_IMG=True # ! set false to get all render images.

BBOX_HEIGHT_CONSTRAINTS=False #  * put it into cmd, False for occluded scene.

## code selection
USE_POSA_ESTIMATE_CAMERA=True
## model setting 
UPDATE_SCENE=True

USE_FILTER_CONF=False
EVALUATE=False

# ! render scan
RENDER_SCAN = True

# constant ratio
CONSTANT_CONTACT_POSA_RATIO=0.6
CONSTANT_CONTACT_DISTANCE=0.05

COMMON_SIZE = {
    5: [0.9 * one for one in [0.61, 0.82, 0.57]],
    6: [0.9 * one for one in [2.36, 0.63, 0.96]], #sofa: 
    7: [0.9 * one for one in [0.65, 0.5, 0.65]],
}

# pytorch3d depth loss
PYTORCH3D_DEPTH=True

CLASS_ID_MAP = {
    "bat": 34,
    "bench": 13,
    "bicycle": 1,
    "car": 2,
    "laptop": 63,
    "motorcycle": 3,
    "skateboard": 36,
    "surfboard": 37,
    "tennis": 38,
    "chair": 56,
    "couch": 57,
    "bed": 59,
    "dining table": 60,
}

MEAN_INTRINSIC_SCALE = {  # Empirical intrinsic scales learned by our method.
    "bat": 0.40,
    "bench": 0.95,
    "bicycle": 0.91,
    "laptop": 0.24,
    "motorcycle": 1.02,
    "skateboard": 0.35,
    "surfboard": 1.0,
    "tennis": 0.33,
    "chair": 0.8,
}

MESH_MAP = {  # Class name -> list of paths to objs.
    "bicycle": ["models/meshes/bicycle_01.obj"],
    "chair": ["/is/cluster/hyi/workspace/HCI/hdsr/mover_ori_repo/models/meshes/20335_Bofinger_Chair_V1_smplify.obj"],
}

PART_LABELS = {
    "person": [("models/meshes/person_labels.json", {})],
    "chair": [
        (
            "models/meshes/chair_bench_labels.json",
            {"seat": ["butt"], "seat back": ["back"]},
        )
    ],
}
INTERACTION_MAPPING = {
    "bat": ["lpalm", "rpalm"],
    "bench": ["back", "butt"],
    "chair": ["butt", "back"],
    "bicycle": ["lhand", "rhand", "butt"],
    "laptop": ["lhand", "rhand"],
    "motorcycle": ["lhand", "rhand", "butt"],
    "skateboard": ["lfoot", "rfoot", "lhand", "rhand"],
    "surfboard": ["lfoot", "rfoot", "lhand", "rhand"],
    "tennis": ["lpalm", "rpalm"],
}
BBOX_EXPANSION = {
    "bat": 0.5,
    "bench": 0.3,
    "bicycle": 0.0,
    "bottle": 0.3,
    "chair": 0.3,
    "couch": 0.3,
    "cup": 0.3,
    "horse": 0.0,
    "laptop": 0.2,
    "motorcycle": 0.0,
    "skateboard": 0.8,
    "surfboard": 0,
    "tennis": 0.4,
    "wineglass": 0.3,
}
BBOX_EXPANSION_PARTS = {
    "bat": 2.5,
    "bench": 0.5,
    "bicycle": 0.7,
    "bottle": 0.3,
    "chair": 0.3,
    "couch": 0.3,
    "cup": 0.3,
    "horse": 0.3,
    "laptop": 0.0,
    "motorcycle": 0.7,
    "skateboard": 0.5,
    "surfboard": 0.2,
    "tennis": 2,
    "wineglass": 0.3,
}
INTERACTION_THRESHOLD = {
    "bat": 5,
    "bench": 3,
    "chair": 3,
    "bicycle": 2,
    "laptop": 2.5,
    "motorcycle": 5,
    "skateboard": 3,
    "surfboard": 5,
    "tennis": 5,
}

NYU40CLASSES = ['void',
                'wall', 'floor', 'cabinet', 'bed', 'chair',
                'sofa', 'table', 'door', 'window', 'bookshelf',
                'picture', 'counter', 'blinds', 'desk', 'shelves',
                'curtain', 'dresser', 'pillow', 'mirror', 'floor_mat',
                'clothes', 'ceiling', 'books', 'refridgerator', 'television',
                'paper', 'towel', 'shower_curtain', 'box', 'whiteboard',
                'person', 'night_stand', 'toilet', 'sink', 'lamp',
                'bathtub', 'bag', 'otherstructure', 'otherfurniture', 'otherprop']

NYU37_TO_PIX3D_CLS_MAPPING = {0:0, 1:0, 2:0, 3:8, 4:1, 5:3, 6:5, 7:6, 8:8, 9:2, 10:2, 11:0, 12:0, 13:2, 14:4,
                              15:2, 16:2, 17:8, 18:0, 19:0, 20:0, 21:0, 22:0, 23:0, 24:8, 25:8, 26:0, 27:0, 28:0,
                              29:8, 30:8, 31:0, 32:8, 33:0, 34:0, 35:0, 36:0, 37:8}
RECON_3D_CLS = [3,4,5,6,7,8,10,14,15,17,24,25,29,30,32]
number_pnts_on_template = 2562
pix3d_n_classes = 9
cls_reg_ratio = 10
obj_cam_ratio = 1

COCO_CLS_NAME = {0: u'__background__',
 1: u'person',
 2: u'bicycle',
 3: u'car',
 4: u'motorcycle',
 5: u'airplane',
 6: u'bus',
 7: u'train',
 8: u'truck',
 9: u'boat',
 10: u'traffic light',
 11: u'fire hydrant',
 12: u'stop sign',
 13: u'parking meter',
 14: u'bench',
 15: u'bird',
 16: u'cat',
 17: u'dog',
 18: u'horse',
 19: u'sheep',
 20: u'cow',
 21: u'elephant',
 22: u'bear',
 23: u'zebra',
 24: u'giraffe',
 25: u'backpack',
 26: u'umbrella',
 27: u'handbag',
 28: u'tie',
 29: u'suitcase',
 30: u'frisbee',
 31: u'skis',
 32: u'snowboard',
 33: u'sports ball',
 34: u'kite',
 35: u'baseball bat',
 36: u'baseball glove',
 37: u'skateboard',
 38: u'surfboard',
 39: u'tennis racket',
 40: u'bottle',
 41: u'wine glass',
 42: u'cup',
 43: u'fork',
 44: u'knife',
 45: u'spoon',
 46: u'bowl',
 47: u'banana',
 48: u'apple',
 49: u'sandwich',
 50: u'orange',
 51: u'broccoli',
 52: u'carrot',
 53: u'hot dog',
 54: u'pizza',
 55: u'donut',
 56: u'cake',
 57: u'chair',
 58: u'couch',
 59: u'potted plant',
 60: u'bed',
 61: u'dining table',
 62: u'toilet',
 63: u'tv',
 64: u'laptop',
 65: u'mouse',
 66: u'remote',
 67: u'keyboard',
 68: u'cell phone',
 69: u'microwave',
 70: u'oven',
 71: u'toaster',
 72: u'sink',
 73: u'refrigerator',
 74: u'book',
 75: u'clock',
 76: u'vase',
 77: u'scissors',
 78: u'teddy bear',
 79: u'hair drier',
 80: u'toothbrush'
}

# in camera view mode
SIZE_FOR_DIFFERENT_CLASS = {
    'chair': [0.4, 1.0, 0.4], 
    'sofa': [2.0, 0.5, 0.6], 
    'table': [2.0, 1.0, 0.6], 
}

KPTS_OPENPOSE_FOR_SMPLX = [
        55, 
        12, 
        17, 
        19, 
        21, 
        16, 
        18, 
        20, 
        0, 
        2, 
        5, 
        8, 
        1, 
        4, 
        7, 
        56, 
        57, 
        58, 
        59, 
        60, 
        61, 
        62, 
        63, 
        64, 
        65, 
    ]

DEFAULT_LOSS_WEIGHTS = {  # Loss weights.
    #################################
    ###### optimize gp.
    #################################
    "debug_10127": { # 2D Cues Only
        "loss_weight": {    
            "lw_scale": 1000, #100 is more near original, pure scene use 100, when introduce human, set it as 10.
            "lw_proj_bbox": 600,
            "lw_offscreen": 1,
            "lw_ground_objs": 0,
            "lw_orientation_penalty": 0.0,
            "lw_collision_objs": 0, #1000,
            "lw_sil": 0.5, 
            "lw_edge": 0.05, #10 is big, should be 5 to make it accurate to the edge
            "lw_gp_contact": 0, 
            "lw_sdf": 0, 
            "lw_depth": 0, #2e-2, #2e1, #2e0, 
            "lw_overlap": 0,
            "lw_contact": 0,
            "lw_contact_coarse": 0,
        },
    },

    "debug_17000": {  
        "loss_weight": {    
            "lw_scale": 1000, #100 is more near original, pure scene use 100, when introduce human, set it as 10.
            "lw_proj_bbox": 1000,
            "lw_offscreen": 1,
            "lw_ground_objs": 0,
            "lw_orientation_penalty": 0.0,
            "lw_collision_objs": 0, #1000,
            "lw_sil": 0.3, 
            "lw_edge": 0.03, #10 is big, should be 5 to make it accurate to the edge
            "lw_gp_contact": 0, 
            "lw_sdf": 1e3, 
            "lw_depth": 8e1, #2e-2, #2e1, #2e0,
            "lw_fs_depth": 8e1, 
            "lw_overlap": 0,
            "lw_contact": 0,
            "lw_contact_coarse": 1e5,
        },
    },
}