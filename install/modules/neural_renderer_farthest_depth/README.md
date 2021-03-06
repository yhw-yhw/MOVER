# About

Here we provide a faster reimplementation of the Neural 3D Mesh Renderer. More specifically we reimplemented the forward pass of the rasterization process.
Everything else is kept the same with the original repo for backwards compatibility.
If you use our optimized version of NMR for your research, consider also citing our work:

```
@Inproceedings{jiang2020mpshape,
Title          = {Coherent Reconstruction of Multiple Humans from a Single Image},
Author         = {Jiang, Wen and Kolotouros, Nikos and Pavlakos, Georgios and Zhou, Xiaowei and Daniilidis, Kostas},
Booktitle      = {CVPR},
Year           = {2020}
}
```

# Neural 3D Mesh Renderer (CVPR 2018)

This repo contains a PyTorch implementation of the paper [Neural 3D Mesh Renderer](http://hiroharu-kato.com/projects_en/neural_renderer_farthest_depth.html) by Hiroharu Kato, Yoshitaka Ushiku, and Tatsuya Harada.
It is a port of the [original Chainer implementation](https://github.com/hiroharu-kato/neural_renderer_farthest_depth) released by the authors.
Currently the API is the same as in the original implementation with some smalls additions (e.g. render using a general 3x4 camera matrix, lens distortion coefficients etc.). However it is possible that it will change in the future.

The library is fully functional and it passes all the test cases supplied by the authors of the original library.
Detailed documentation will be added in the near future.

## Requirements
Python 2.7+ and PyTorch>=0.4.0.

The code has been tested only with PyTorch 0.4.0, there are no guarantees that it is compatible with older versions.
Currently the library has both Python 3 and Python 2 support.

**Note**: In some newer PyTorch versions you might see some compilation errors involving AT_ASSERT. In these cases you can use the version of the code that is in the branch *at_assert_fix*. These changes will be merged into master in the near future.
## Installation
You can install the package by running
```
python setup.py install
```
Since running install.py requires PyTorch, make sure to install PyTorch before running the above command.
## Running examples
```
python ./examples/example1.py
python ./examples/example2.py
python ./examples/example3.py
python ./examples/example4.py
```


## Example 1: Drawing an object from multiple viewpoints

![](https://raw.githubusercontent.com/hiroharu-kato/neural_renderer_farthest_depth/master/examples/data/example1.gif)

## Example 2: Optimizing vertices

Transforming the silhouette of a teapot into a rectangle. The loss function is the difference between the rendered image and the reference image.

Reference image, optimization, and the result.

![](https://raw.githubusercontent.com/hiroharu-kato/neural_renderer_farthest_depth/master/examples/data/example2_ref.png) ![](https://raw.githubusercontent.com/hiroharu-kato/neural_renderer_farthest_depth/master/examples/data/example2_optimization.gif) ![](https://raw.githubusercontent.com/hiroharu-kato/neural_renderer_farthest_depth/master/examples/data/example2_result.gif)

## Example 3: Optimizing textures

Matching the color of a teapot with a reference image.

Reference image, result.

![](https://raw.githubusercontent.com/hiroharu-kato/neural_renderer_farthest_depth/master/examples/data/example3_ref.png) ![](https://raw.githubusercontent.com/hiroharu-kato/neural_renderer_farthest_depth/master/examples/data/example3_result.gif)

## Example 4: Finding camera parameters

The derivative of images with respect to camera pose can be computed through this renderer. In this example the position of the camera is optimized by gradient descent.

From left to right: reference image, initial state, and optimization process.

![](https://raw.githubusercontent.com/hiroharu-kato/neural_renderer_farthest_depth/master/examples/data/example4_ref.png) ![](https://raw.githubusercontent.com/hiroharu-kato/neural_renderer_farthest_depth/master/examples/data/example4_init.png) ![](https://raw.githubusercontent.com/hiroharu-kato/neural_renderer_farthest_depth/master/examples/data/example4_result.gif)


## Citation

```
@InProceedings{kato2018renderer
    title={Neural 3D Mesh Renderer},
    author={Kato, Hiroharu and Ushiku, Yoshitaka and Harada, Tatsuya},
    booktitle={The IEEE Conference on Computer Vision and Pattern Recognition (CVPR)},
    year={2018}
}
```
