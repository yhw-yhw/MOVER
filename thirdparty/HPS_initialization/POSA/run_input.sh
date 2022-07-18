#!/bin/bash
# input $1: the directory contains the pkl files which stores the smplx parameters.
pkl_file_path=$1/split
# pkl_file_path=$2/split
rand_samples_dir=$1/posa_render_results
save_dir=$1/posa_contact_npy_newBottom
python src/gen_rand_samples.py \
    --config cfg_files/contact.yaml \
    --checkpoint_path $POSA_dir/trained_models/contact.pt \
    --pkl_file_path ${pkl_file_path} \
    --render 1 \
    --num_rand_samples 1 \
    --save 1 \
    --rand_samples_dir ${rand_samples_dir} \
    --save_dir ${save_dir}
