# Demo dataset
We provide three videos for the demo, i.e., MPH8_00034_01 from PROX qualitative, 3bHallway_mati2_2014-04-30-22-46-22 from piGraph and PROX quantitative.

Each directory contains:
```
--prox_scans_pre: ground truth 3D scans (option, and it is only used for visualization)
```

1. scene intialization.  

--Total3D_label_occNet_results: scene initialization from [Total3DUnderstanding](https://github.com/yinyunie/Total3DUnderstanding) and [OccupancyNet](https://github.com/autonomousvision/occupancy_networks).

2. HPS.

```
--mv_smplifyx_input_OneEuroFilter_PARE_PARE3DJointOneConfidence_OP2DJoints: batch-wise smplifyx input
--smplifyx_results_st0: batch-wise smplify results under a perspective camera with no orentation.                                        
--smplifyx_results_st0_camera_gp_posaVersion_meanFeetAsInit: optimize ground plane and camera orientation with contacted feet vertices.
--smplifyx_results_st2_newCamera_gpConstraints_posaVersion: optimize new human pose and shape (HPS) under new camera and ground plane constraints.  
```
3. Final Optimized Results. 

```
--refine_results: Human-scene interactions and 2D cues refined 3D scene results.
```

# MOVER results

We provide MOVER results for PROX qualitative dataset.

The optimized results will be: 
```
refine_results/scene_reconstruction_s3kind1/obj_-1: 
    --model_scene_0_lr0.002_end: optimized results with only 2D cues.
    --model_scene_1_lr0.002_end: optimized results with HSIs and 2D cues.
    --scene_body_end: rendering results.
```