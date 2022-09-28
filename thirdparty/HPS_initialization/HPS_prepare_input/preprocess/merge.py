import os
import sys
import argparse

sys.path.append('preprocess/footskate_reducer')
sys.path.append('preprocess/footskate_reducer/ground_detector')
from footskate_reducer.ground_detector.op_filter_json_merge import main0
from demo_merge import main1
from demo_pare_result_merge import main2
from merge_parser import parser_pare, parser_pare_result, parser_op_filter


def main(args):
    video_name = args.video_name
    input_dir = args.input_dir
    out_dir = args.out_dir
    video_path = os.path.join(input_dir, video_name, video_name + ".mp4")
    openpose_dir = os.path.join(input_dir, video_name, "openpose")
    image_dir = os.path.join(input_dir, video_name, "images")

    # step0: openpose filter
    parser0 = parser_op_filter()
    parser0.set_defaults(root=openpose_dir,
                         dump=os.path.join(out_dir, video_name, "openpose_OneEurofilter"),
                         img_dir=image_dir,
                         viz=True)
    args0 = parser0.parse_args([])
    main0(args0)

    # step1: pare
    parser1 = parser_pare()
    pare_model = '../hrnet_model'
    pare_exp = 'pare'
    parser1.set_defaults(cfg=f'{pare_model}/config.yaml', ckpt=f'{pare_model}/checkpoint.ckpt',
                         output_folder=os.path.join(out_dir, video_name), vid_file=video_path, draw_keypoints=True,
                         detector='maskrcnn', exp=pare_exp)
    args1 = parser1.parse_args([])
    main1(args1)

    # step2: op2smplifyx_withPARE
    parser2 = parser_pare_result()
    output_folder = os.path.join(out_dir, video_name, "op2smplifyx_withPARE")
    json_folder = os.path.join(out_dir, video_name, "openpose_OneEurofilter")
    pare_result = os.path.join(out_dir, video_name, pare_exp, "pare_output.pkl")
    cam_dir = "../smplifyx_cam"
    model_folder = 'data/body_models'
    vposer_folder = '../smplifyx-file/vposer_v1_0'
    segm_fn_path = '../smplifyx-file/smplx_parts_segm.pkl'

    parser2.set_defaults(config='cfg_files/fit_smpl.yaml', export_mesh=True, save_new_json=True,
                         json_folder=json_folder, data_folder=image_dir, output_folder=output_folder,
                         pare_result=pare_result, cam_dir=cam_dir, visualize=False,
                         model_folder=model_folder, vposer_ckpt=vposer_folder,
                         part_segm_fn=segm_fn_path, gender='male',
                         check_inverse_feet=False)
    args2 = parser2.parse_args([])
    args_dict2 = vars(args2)
    main2(**args_dict2)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='merge preprocess for smplifyx-modified')
    parser.add_argument('--video_name', type=str, default='test_video1',
                        help='video name without suffix in input dir to process')
    parser.add_argument('--out_dir', type=str, default='/share/wenzhuoliu/code/mover-lwz/out-data',
                        help='output dir')
    parser.add_argument('--input_dir', type=str, default='/share/wenzhuoliu/code/mover-lwz/input-data',
                        help='input dir')
    args = parser.parse_args()
    main(args)
