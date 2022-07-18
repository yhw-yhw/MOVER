# Requirements
* [detectron2](https://github.com/facebookresearch/detectron2)
* [Total3DUnderstanding](https://github.com/yinyunie/Total3DUnderstanding)
* [OccupancyNet](https://github.com/autonomousvision/occupancy_networks)
* [Manifold](https://github.com/hjwdzh/Manifold)

# Run PointRend

Run all images in ${INPUT_IMAGE_DIR}

```
./detection/run_pointrend_coco.sh ${INPUT_IMAGE_DIR} -1
```

Run specific image frame ${i} in  ${INPUT_IMAGE_DIR}

```
./detection/run_pointrend_coco.sh ${INPUT_IMAGE_DIR} ${i}
```

# Test Total3D

Get a initial 3D reconstructed scene from one single RGB image, which consists of multiple mesh objects.
<!-- /is/cluster/hyi/workspace/HCI/Total3DUnderstanding/run.sh  -->

```
./Total3DUnderstanding/run.sh
```

|            Total3D Results             |                 Occupancy Results                 |
|:--------------------------------:|:------------------------------------------:|
| ![Apple](../../assets/Total3D_results/snapshot00.png) | ![Binoculars](../../assets/Total3D_results/snapshot01.png) |
| ![Apple](../../assets/Total3D_results/snapshot02.png) | ![Binoculars](../../assets/Total3D_results/snapshot03.png) |
| ![Apple](../../assets/Total3D_results/snapshot04.png) | ![Binoculars](../../assets/Total3D_results/snapshot05.png) |
| ![Apple](../../assets/Total3D_results/snapshot06.png) | ![Binoculars](../../assets/Total3D_results/snapshot07.png) |
| ![Apple](../../assets/Total3D_results/snapshot08.png) | ![Binoculars](../../assets/Total3D_results/snapshot09.png) |
| ![Apple](../../assets/Total3D_results/snapshot10.png) | ![Binoculars](../../assets/Total3D_results/snapshot11.png) |


# Run OccupancyNet 
Transfer each reconstructed object from Total3DUnderstanding into a water-tight mesh. 

And it also contains the [Manifold/build/simplify](git://github.com/hjwdzh/Manifold) to smplify the water-tight mesh.
please follow the [Manifold/README.md](thirdparty/scene_initialization/Manifold) to build.

```
<!-- /is/cluster/hyi/workspace/Multi-IOI/occupancy_networks/run_label.sh  -->
./occupancy_networks/run_label.sh
```


# Comparison with Implicit3DUnderstanding

We also test 
```
https://github.com/chengzhag/Implicit3DUnderstanding
```
on PROX datset. It does not work well.

|            MPH8             |                 N3OpenArea                 |
|:--------------------------------:|:------------------------------------------:|
| ![MPH8](../../assets/Implicit3DUnderstanding_results/MPH8.jpeg) | ![N3OpenArea](../../assets/Implicit3DUnderstanding_results/N3OpenArea2.jpeg) |
|  ![MPH8](../../assets/Implicit3DUnderstanding_results/MPH8_results.png)   |  ![N3OpenArea](../../assets/Implicit3DUnderstanding_results/N3OpenArea2_results.png)   |