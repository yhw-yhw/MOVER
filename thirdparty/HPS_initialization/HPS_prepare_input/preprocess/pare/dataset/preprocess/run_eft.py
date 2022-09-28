import cv2
import torch

from ...utils.vibe_image_utils import get_single_image_crop_demo


def run_eft_step(eft_fitter, img_file, keypoints, bbox, crop_size=224, counter=0, debug=True, device='cuda'):
    img = cv2.cvtColor(cv2.imread(img_file), cv2.COLOR_BGR2RGB)

    norm_img, raw_img, kp_2d = get_single_image_crop_demo(
        img,
        bbox,
        kp_2d=keypoints,
        scale=1.0,
        crop_size=crop_size,
    )

    kp_2d[:, :-1] = 2. * kp_2d[:, :-1] / crop_size - 1.
    kp_2d = torch.from_numpy(kp_2d)
    batch = {'img': norm_img.unsqueeze(0), 'keypoints': kp_2d.unsqueeze(0)}

    batch = {k: v.to(device) if isinstance(v, torch.Tensor) else v for k, v in batch.items()}
    batch = {k: v.float() if isinstance(v, torch.Tensor) else v for k, v in batch.items()}
    output = eft_fitter.finetune_step(batch, batch_nb=counter)

    if debug:
        batch['disp_img'] = batch['img']
        # This saves the EFT renderings to hparams.LOG_DIR directory
        eft_fitter.visualize_results(batch, output, counter, has_error_metrics=False)

    return output