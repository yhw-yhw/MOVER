import os
import json
from tqdm import tqdm
from argparse import ArgumentParser

import cv2
import math
import pickle

import torch, torchvision
print(torch.__version__, torch.cuda.is_available())
assert torch.__version__.startswith("1.7")

import detectron2
from detectron2.utils.logger import setup_logger
setup_logger()

# import some common libraries
import numpy as np
import cv2
import torch
# from google.colab.patches import cv2_imshow

# import some common detectron2 utilities
from detectron2 import model_zoo
from detectron2.engine import DefaultPredictor
from detectron2.config import get_cfg
from detectron2.utils.visualizer import Visualizer, ColorMode
from detectron2.data import MetadataCatalog
coco_metadata = MetadataCatalog.get("coco_2017_val")

from PIL import Image
# import PointRend project
from detectron2.projects import point_rend

detectron2_path = './' 
cfg = get_cfg()
# Add PointRend-specific config
point_rend.add_pointrend_config(cfg)
# Load a config from file
# cfg.merge_from_file(os.path.join(detectron2_path, "projects/PointRend/configs/InstanceSegmentation/pointrend_rcnn_R_50_FPN_3x_coco.yaml"))
cfg.merge_from_file(os.path.join(detectron2_path, "PointRend/configs/InstanceSegmentation/pointrend_rcnn_X_101_32x8d_FPN_3x_coco.yaml"))
# cfg.merge_from_file(os.path.join(detectron2_path, "projects/PointRend/configs/InstanceSegmentation/pointrend_rcnn_R_101_FPN_3x_coco.yaml"))
cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.30  # set threshold for this model
# Use a model from PointRend model zoo: https://github.com/facebookresearch/detectron2/tree/master/projects/PointRend#pretrained-models
# cfg.MODEL.WEIGHTS = "detectron2://PointRend/InstanceSegmentation/pointrend_rcnn_R_50_FPN_3x_coco/164955410/model_final_3c3198.pkl"
cfg.MODEL.WEIGHTS = "detectron2://PointRend/InstanceSegmentation/pointrend_rcnn_X_101_32x8d_FPN_3x_coco/28119989/model_final_ba17b9.pkl"
# cfg.MODEL.WEIGHTS = "detectron2://PointRend/InstanceSegmentation/pointrend_rcnn_R_101_FPN_3x_coco/28119983/model_final_3f4d2a.pkl"
predictor = DefaultPredictor(cfg)

def inference_detector(det_model, img_fn):
        
    im = cv2.imread(img_fn)
    # cfg = get_cfg()
    # cfg.merge_from_file(model_zoo.get_config_file("COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_3x.yaml"))
    # cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.5  # set threshold for this model
    # cfg.MODEL.WEIGHTS = model_zoo.get_checkpoint_url("COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_3x.yaml")
    # mask_rcnn_predictor = DefaultPredictor(cfg)
    # mask_rcnn_outputs = mask_rcnn_predictor(im)
    outputs = predictor(im)
    return outputs["instances"]


def show_result_pyplot(model, img, result, score_thr=0.3, fig_size=(15, 10), save_path=None, viz=True):
    """Visualize the detection results on the image.

    Args:
        model (nn.Module): The loaded detector.
        img (str or np.ndarray): Image filename or loaded image.
        result (tuple[list] or list): The detection result, can be either
            (bbox, segm) or just bbox.
        score_thr (float): The threshold to visualize the bboxes and masks.
        fig_size (tuple): Figure size of the pyplot figure.
    """
    if hasattr(model, 'module'):
        model = model.module
    img = model.show_result(img, result, score_thr=score_thr, show=False)
    if save_path is not None:
        mmcv.imwrite(img, save_path)
    if viz:
        plt.figure(figsize=fig_size)
        plt.imshow(mmcv.bgr2rgb(img))
        plt.show()

coco_cls_name = {0: u'__background__',
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
 58: u'sofa', #'couch',
 59: u'potted plant',
 60: u'bed',
 61: u'table', #'dining table',
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


sunrgb_labels_dict = {
  0: 'wall',
  1: 'floor',
  2: 'cabinet',
  3:  'bed',
  4:  'chair',
  5:  'sofa',
  6:  'table', 
  7:  'door',
  8:  'window',
  9:  'bookshelf',
  10:  'picture',
  11:  'counter',
  12:  'blinds',
  13:  'desk',
  14:  'shelves',
  15:  'curtain',
  16:  'dresser',
  17:  'pillow',
  18:  'mirror',
  19:  'floor_mat',
  20:  'clothes',
  21:  'ceiling',
  22:  'books',
  23:  'fridge',
  24:  'tv',
  25:  'paper',
  26:  'towel',
  27:  'shower_curtain',
  28:  'box',
  29:  'whiteboard',
  30:  'person',
  31:  'night_stand',
  32:  'toilet',
  33:  'sink',
  34:  'lamp',
  35:  'bathtub',
  36:  'bag',
}


ori_useful_idx = [1, 57, 58, 60, 61]
# detectron2
# ori_useful_idx = [one-1 for one in ori_useful_idx]

def main():
    """Visualize the demo images.

    Using mmdet to detect the human.
    """
    parser = ArgumentParser()

    parser.add_argument('--img-root', type=str, default='', help='Image root')
    parser.add_argument('--out-img-root', type=str, default='', help='root of the output img file.')

    parser.add_argument('--height', type=int, default=530, help='image idx in the folder')
    parser.add_argument('--width', type=int, default=730, help='image idx in the folder')

    parser.add_argument('--show', action='store_true', default=False, help='whether to show img')
    parser.add_argument('--save', action='store_true', default=False, help='whether to save img results')

    parser.add_argument('--device', default='cuda:0', help='Device used for inference')
    parser.add_argument('--bbox-thr', type=float, default=0.3, help='Bounding bbox score threshold')
    parser.add_argument('--kpt-thr', type=float, default=0.3, help='Keypoint score threshold')

    parser.add_argument('--idx', type=int, default=-1, help='image idx in the folder')
    
    args = parser.parse_args()

    assert args.show or (args.out_img_root != '')

    img_idx = args.idx
    # image_name = os.path.join(args.img_root, args.img)
    image_names = sorted([os.path.join(args.img_root, x) for x in os.listdir(args.img_root)
                   if x.endswith('.png') or x.endswith('.jpg') or x.endswith('.jpeg')])

    if img_idx == -1:
        img_idx_list = range(len(image_names))
    else:
        img_idx_list = [img_idx]

    for img_idx in img_idx_list:
        image_name = image_names[img_idx]

        redirect_img_name = image_name.replace(args.img_root, args.out_img_root)
        
        # rescale
        os.makedirs(args.out_img_root, exist_ok=True)
        width, height = args.width, args.height
        img = cv2.imread(image_name)
        scale = get_scale(img, width, height)
        # import ipdb; ipdb.set_trace()
        s_img = scale_image(img, scale)
        c_img = crop_image(s_img, width, height)
        cv2.imwrite(redirect_img_name, c_img)

        det_results = inference_detector(predictor, redirect_img_name)

        out_dir = os.path.join(args.out_img_root, os.path.splitext(image_name.split("/")[-1])[0])
        if args.show:
            os.makedirs(out_dir, exist_ok=True)
            out_det_file = os.path.join(out_dir, f'detections.png')

            v = Visualizer(img[:, :, ::-1], coco_metadata, scale=1.2, instance_mode=ColorMode.IMAGE_BW)
            point_rend_result = v.draw_instance_predictions(det_results.to("cpu")).get_image()
            Image.fromarray(point_rend_result).save(out_det_file)
            Image.fromarray(point_rend_result).save(os.path.join(out_dir, f'../{img_idx:06d}_det.png'))
            # cv2.imwrite(out_det_file, point_rend_result)

        os.remove(redirect_img_name)
        # import pdb; pdb.set_trace()

        predictions = det_results
        boxes = predictions.pred_boxes.tensor.cpu().numpy() if predictions.has("pred_boxes") else None
        scores = predictions.scores.cpu().numpy() if predictions.has("scores") else None
        classes = predictions.pred_classes.cpu().numpy() if predictions.has("pred_classes") else None
        masks = np.asarray(predictions.pred_masks.cpu())


        out_file = os.path.join(out_dir, f'detections.json')
        out_pkl_file = os.path.join(out_dir, f'detections.pkl')
        ori_out_pkl_file = os.path.join(out_dir, f'ori_detections.pkl')
        img_file = os.path.join(out_dir, f'img.jpg')
        cv2.imwrite(img_file, c_img)

        useful_idx = [one-1 for one in ori_useful_idx]
        
        # output json;
        result = []
        for idx in range(classes.shape[0]):
            cls_id = classes[idx]
            if cls_id not in useful_idx:
                continue
            else:

                if coco_cls_name[classes[idx]+1] in sunrgb_labels_dict.values():
                    tmp_dict = {'class': coco_cls_name[classes[idx]+1]}
                    if scores[idx] > args.bbox_thr:
                        # import pdb;pdb.set_trace()
                        tmp_dict['bbox'] = np.concatenate((boxes[idx], scores[idx:idx+1])).tolist()
                        tmp_dict['mask'] = masks[idx].tolist()
                    else:
                        continue
                    result.append(tmp_dict)

        with open(out_file, 'w') as f:
            json.dump(result, f)
        

def get_scale(img, width, height): 
    resize_scale = 1.
    h_scale = 0
    w_scale = 0       
    height_scale = float(height) / img.shape[0] # H, W, 3
    width_scale = float(width) / img.shape[1]
    if height_scale > h_scale:
        h_scale = height_scale
    if width_scale > w_scale:
        w_scale = width_scale
    if h_scale > 1 or w_scale > 1:
        print ("max_h, max_w should < W and H!")
        exit(-1)
    resize_scale = h_scale
    if w_scale > h_scale:
        resize_scale = w_scale
    return resize_scale

def scale_image(image, scale=1, interpolation='linear'):
    """ resize image using cv2 """
    if interpolation == 'linear':
        return cv2.resize(image, None, fx=scale, fy=scale, interpolation=cv2.INTER_LINEAR)
    if interpolation == 'nearest':
        return cv2.resize(image, None, fx=scale, fy=scale, interpolation=cv2.INTER_NEAREST)

def crop_image(image, width, height):
    h, w = image.shape[0:2]
    new_h = h
    new_w = w
    if new_h > height:
        new_h = height
    else:
        new_h = h
    if new_w > width:
        new_w = width
    else:
        new_w = w
    start_h = int(math.ceil((h - new_h) / 2))
    start_w = int(math.ceil((w - new_w) / 2))
    finish_h = start_h + new_h
    finish_w = start_w + new_w
    
    return image[start_h:finish_h, start_w:finish_w]
    
if __name__ == '__main__':
    main()
