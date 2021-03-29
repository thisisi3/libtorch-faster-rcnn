

## Introduction

When it comes to building deep learning models in C++, libtorch is a good choice. Libtorch and Pytorch are essentially the same, libtorch provides the C++ interface while Pytorch provides the Python interface. Libtorch's interface is designed to be almost the same as Python's, which makes converting Pytorch code to C++ code very easy. One can literally translate Pytorch code to libtorch code line by line. What's more, one can train a model using Pytorch and use the model in C++ environment. 

In this project, I use libtorch to implement the classic object detection model Faster RCNN. I hope to give one sense of how one can convert a Pytorch model to a C++ model in aspects of both train and inference. The overall structure and configuration very much follows mmdetection(v2.3.0)'s implementation of Faster RCNN.   [MMdetection](https://github.com/open-mmlab/mmdetection/tree/v2.3.0) is a well known object detection framework that implements many of the popular object detection models. 



## Compile and Use

#### requirements

- opencv v4.1.4
- torchvision 0.7.0
- libtorch v1.6.0 release (most likely compatible with other libtorch versions but not tested)

#### torchvision

As we know pytorch does not come with CV operators like nms, roi_align etc. We need torchvision's C++ implementation. Here I only took code of nms, roi_align and roi_pool, and put them in `cvops/` to compile together with the project.

#### compile with cmake(v3.19.2)

```shell
mkdir build
cd build
cmake ..
cmake --build . --config Release --parallel 8
```

#### train

```shell
./build/train configs/faster_rcnn_r50_fpn_1x_voc.json --work-dir work_dir --gpu 0
```

#### inference

```shell
./build/test faster_rcnn_r50_fpn_1x_voc.json epoch_12.pt --out epoch_12.bbox.json --gpu 0
```

#### ImageNet pretrained backbones 

For pytorch users, they usually go to torchvision for ImageNet pretrained weights. But those weights can not be directly loaded in libtorch. One way is to use torchscript to load model in C++, check out the [official tutorial](https://pytorch.org/tutorials/advanced/cpp_export.html) for more details. Here is what I did:

```python
img_tsr = torch.rand(1, 3, 1000, 600)
model = torchvision.models.resnet50(True)
model.eval()
traced = torch.jit.trace(model, img_tsr)
traced.save('resnet50.pt')
```

**IMPORTANT:** If you use trace method, do remember to run `eval()` first, as trace will change tracked means and stds in BN layers. 



## Benchmark

Train: **voc2007-trainval**

Test: **voc2007-test**

|       | backbone | mAP   | AP50  |
| ----- | -------- | ----- | ----- |
| mmdet | Resnet50 | 0.437 | 0.769 |
| this  | Resnet50 | 0.438 | 0.768 |

VOC dataset is used as the main dataset due to limited GPU resource. But the metrics are all using coco's. Notice that there's around 0.02 variance in mAP among different trains. VOC's XML annotations are first converted to coco format and only non-difficult bboxes are used for train and test. 


