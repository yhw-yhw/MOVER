# Requirements
* [SMPLify-X](https://smpl-x.is.tue.mpg.de/)
* [POSA](https://github.com/mohamedhassanmus/POSA)


We provide the codebase for 

* Prepare the input. [TODO]

* Batch-wise SMPLify-X under a normal perspective camera with Identity Rotation Matrix

* Run POSA to get the contact vertices for each human.

* Optimize the camera pose and the height of the ground plane with contacted feet. [TODO]

* Optimize Batch-wise SMPLify-X under new optimized camera pose and ground plane constraints. [TODO]

# Run Batch-wise SMPLify-X

```
bash run_video.sh ${save_dir}/${video_name} ${video_length}
```

# Run POSA on Estimated HPS 

Requirements: Install POSA following https://github.com/mohamedhassanmus/POSA

## Split one whole pkl into seperate frames results
```
cd pre_POSA
bash run.sh ${save_dir}/${video_name}/smplifyx_results_st0/results/001_all.pkl
```

## Run POSA on each frame result.

```
cd POSA
bash run_input.sh ${save_dir}/${video_name}/smplifyx_results_st0/results
```

# Estimated Camera Poses and Ground Plane 


# Re-run Batch-wise SMPLify-X under New Camera Poses and Ground Plane Constraints
