# Paper Reproduction: Image Representation on Curved Optimal Triangulation

This repository is a reproduction of the paper [*Image Representation on Curved Optimal Triangulation*](https://www.researchgate.net/publication/359340562_Image_Representation_on_Curved_Optimal_Triangulation)

> Abstract: Image triangulation aims to generate an optimal partition with triangular elements to represent the given image. One bottleneck
in ensuring approximation quality between the original image and a piecewise approximation over the triangulation is the
inaccurate alignment of straight edges to the curved features. In this paper, we propose a novel variational method called curved
optimal triangulation, where not all edges are straight segments, but may also be quadratic Bézier curves. The energy function is
defined as the total approximation error determined by vertex locations, connectivity and bending of edges. The gradient formulas
of this function are derived explicitly in closed form to optimize the energy function efficiently. We test our method on several
models to demonstrate its efficacy and ability in preserving features. We also explore its applications in the automatic generation
of stylization and Lowpoly images. With the same number of vertices, our curved optimal triangulation method generates more
accurate and visually pleasing results compared with previous methods that only use straight segments.

![1](./imageopt2/[rasterization]%20imageResult4.jpg)
![1](./imageopt2/[main2]%20vertexGradient%20Image4.jpg)
![3](./imageopt2/erciyuan/[rasterization]%20imageResult2.jpg)
![3](./imageopt2/erciyuan/[main2]%20vertexGradient%20Image2.jpg)
![2](./imageopt2/siyecao/[rasterization]%20imageResult9.jpg)
![2](./imageopt2/siyecao/[main2]%20vertexGradient%20Image9.jpg)

## Environment

- OpenCV 4.6.0
- opencv_contrib-4.6.0
- [CDT](https://github.com/artem-ogre/CDT)
- Eigen 3

## Setup

- Setup environment variables for dependencies : "OPENCV_DIR" "EIGEN3_INCLUDE_DIR" "CDT_DIR"
- Open "./opencv_test/opencv_properties_sheet.props" in Visual Studio and change C/C++ Additional Include Directories to include dependencies
  - `$(OPENCV_DIR)\..\..\include`
  - `D:\code\CDT\CDT\include` -> `${CDT_DIR}\include`
    - i forgot to change this so you need to change it manually. =_=
  - `$(EIGEN3_INCLUDE_DIR)`
- Run `./imageopt_linear/4.cpp`

## Usage for different projects

- `imageopt_const` & `imageopt2`: image triangulation for const color
- `imageopt_linear` : image triangulation for linear color
- `bz_test` & `opencv_test`: some tests of using cdt & opencv
- `RMSEcal`: Calculate the RMSE of two images

## Known Issues

- The constraints on control points and curves during optimization are not strict, resulting in incorrect image filling results