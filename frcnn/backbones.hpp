#ifndef BACKBONES_HPP
#define BACKBONES_HPP

#include <utils.hpp>
#include <torch/torch.h>
#include <json.hpp>

namespace frcnn
{
  using json = nlohmann::json;
  
  /**
     Only support ResNet/ResNext series backbones, the implementation is borrowed from 
     torchvision v0.6.0 with some modifications for detection.
   */
  
  // define Backbone class 
  struct Backbone : torch::nn::Module
  {
    virtual std::vector<torch::Tensor> forward(torch::Tensor x){
      throw std::runtime_error("Please implement backbone's forward method.");
    }
  };

  // freeze a module by setting requires_grad = False
  void freeze_module(torch::nn::Module *ptr);
  
  
  template <typename Block>
  struct ResNetImpl;

  // 3x3 convolution with padding
  torch::nn::Conv2d conv3x3(int64_t in,
			    int64_t out,
			    int64_t stride=1,
			    int64_t groups=1);
  
  // 1x1 convolution
  torch::nn::Conv2d conv1x1(int64_t in, int64_t out, int64_t stride=1);

  struct BasicBlock : torch::nn::Module {
    template <typename Block>
    friend struct ResNetImpl;

    int64_t stride;
    torch::nn::Sequential downsample;
    torch::nn::Conv2d conv1{nullptr}, conv2{nullptr};
    torch::nn::BatchNorm2d bn1{nullptr}, bn2{nullptr};

    static int expansion;

    BasicBlock(int64_t inplaces,
	       int64_t places,
	       int64_t stride = 1,
	       torch::nn::Sequential downsample = nullptr,
	       int64_t groups = 1,
	       int64_t base_width = 64);

    torch::Tensor forward(torch::Tensor x);
  }; // BasicBlock

  struct Bottleneck : torch::nn::Module{
    template <typename Block>
    friend struct ResNetImpl;

    int64_t stride;
    torch::nn::Sequential downsample;
    torch::nn::Conv2d conv1{nullptr}, conv2{nullptr}, conv3{nullptr};
    torch::nn::BatchNorm2d bn1{nullptr}, bn2{nullptr}, bn3{nullptr};

    static int expansion;

    Bottleneck(int64_t inplanes,
	       int64_t planes,
	       int64_t stride = 1,
	       torch::nn::Sequential downsample = nullptr,
	       int64_t groups = 1,
	       int64_t base_width = 64);
    torch::Tensor forward(torch::Tensor x);
  }; // Bottleneck

  template <typename Block>
  struct ResNetImpl : Backbone{
    int64_t groups, base_width, inplanes, frozen_stages;
    torch::nn::Conv2d conv1;
    torch::nn::BatchNorm2d bn1;
    torch::nn::Sequential layer1, layer2, layer3, layer4;

    torch::nn::Sequential _make_layer(int64_t planes,
				      int64_t blocks,
				      int64_t stride = 1);

    ResNetImpl(const std::vector<int>& layers,
	       bool zero_init_residual = false,
	       int64_t groups = 1,
	       int64_t width_per_groups = 64,
	       int64_t frozen_stages = 1);

    std::vector<torch::Tensor> forward(torch::Tensor x) override;

    void freeze_stages(int stages);
    void train(bool on=true) override;

  }; // ResNetImpl

  template <typename Block>
  torch::nn::Sequential ResNetImpl<Block>::_make_layer(int64_t planes,
						       int64_t blocks,
						       int64_t stride){
    torch::nn::Sequential downsample = nullptr;
    if (stride != 1 || inplanes != planes * Block::expansion){
      downsample = torch::nn::Sequential(conv1x1(inplanes, planes * Block::expansion, stride),
					 torch::nn::BatchNorm2d(planes * Block::expansion));
    }

    torch::nn::Sequential layers;
    layers->push_back(Block(inplanes, planes, stride, downsample, groups, base_width));
    inplanes = planes * Block::expansion;

    for(int i=1; i<blocks; ++i){
      layers->push_back(Block(inplanes, planes, 1, nullptr, groups, base_width));
    }
    return layers;
  }

  template <typename Block>
  ResNetImpl<Block>::ResNetImpl(const std::vector<int>& layers,
				bool zero_init_residual,
				int64_t groups,
				int64_t width_per_group,
				int64_t frozen_stages)
    : groups(groups),
      base_width(width_per_group),
      inplanes(64),
      frozen_stages(frozen_stages),
      conv1(torch::nn::Conv2dOptions(3, 64, 7).stride(2).padding(3).bias(false)),
      bn1(64),
      layer1(_make_layer(64, layers[0])),
      layer2(_make_layer(128, layers[1], 2)),
      layer3(_make_layer(256, layers[2], 2)),
      layer4(_make_layer(512, layers[3], 2)) {
    register_module("conv1", conv1);
    register_module("bn1", bn1);
    
    register_module("layer1", layer1);
    register_module("layer2", layer2);
    register_module("layer3", layer3);
    register_module("layer4", layer4);
    
    for (auto& module : modules(/*include_self=*/false)) {
      if (auto M = dynamic_cast<torch::nn::Conv2dImpl*>(module.get()))
	torch::nn::init::kaiming_normal_(M->weight,
					 /*a=*/0,
					 torch::kFanOut,
					 torch::kReLU);
      else if (auto M = dynamic_cast<torch::nn::BatchNorm2dImpl*>(module.get())) {
	torch::nn::init::constant_(M->weight, 1);
	torch::nn::init::constant_(M->bias, 0);
      }
    }

    // Zero-initialize the last BN in each residual branch, so that the residual
    // branch starts with zeros, and each residual block behaves like an
    // identity. This improves the model by 0.2~0.3% according to
    // https://arxiv.org/abs/1706.02677
    if (zero_init_residual)
      for (auto& module : modules(/*include_self=*/false)) {
	if (auto* M = dynamic_cast<Bottleneck*>(module.get()))
	  torch::nn::init::constant_(M->bn3->weight, 0);
	else if (auto* M = dynamic_cast<BasicBlock*>(module.get()))
	  torch::nn::init::constant_(M->bn2->weight, 0);
      }
  }

  template <typename Block>
  std::vector<torch::Tensor> ResNetImpl<Block>::forward(torch::Tensor x) {
    x = conv1->forward(x);
    x = bn1->forward(x).relu_();
    x = torch::max_pool2d(x, 3, 2, 1);

    std::vector<torch::Tensor> feats;
    x = layer1->forward(x);
    feats.push_back(x);
    x = layer2->forward(x);
    feats.push_back(x);
    x = layer3->forward(x);
    feats.push_back(x);
    x = layer4->forward(x);
    feats.push_back(x);
    
    return feats;
  }
  
  // define resnet and resnext for detection, only make frozen_stages configurable 
  struct ResNet18Impl : ResNetImpl<BasicBlock> {
    ResNet18Impl(int frozen_stages=1);
  };
  
  struct ResNet34Impl : ResNetImpl<BasicBlock> {
    ResNet34Impl(int frozen_stages=1);
  };
  
  struct ResNet50Impl : ResNetImpl<Bottleneck> {
    ResNet50Impl(int frozen_stages=1);
  };
  
  struct ResNet101Impl : ResNetImpl<Bottleneck> {
    ResNet101Impl(int frozen_stages=1);
  };
  
  struct ResNet152Impl : ResNetImpl<Bottleneck> {
    ResNet152Impl(int frozen_stages=1);
  };
  
  struct ResNext50_32x4dImpl : ResNetImpl<Bottleneck> {
    ResNext50_32x4dImpl(int frozen_stages=1);
  };
  
  struct ResNext101_32x8dImpl : ResNetImpl<Bottleneck> {
    ResNext101_32x8dImpl(int frozen_stages=1);
  };
  
  
  template <typename Block>
  struct ResNet : torch::nn::ModuleHolder<ResNetImpl<Block>> {
    using torch::nn::ModuleHolder<ResNetImpl<Block>>::ModuleHolder;
  };

  TORCH_MODULE(ResNet18);
  TORCH_MODULE(ResNet34);
  TORCH_MODULE(ResNet50);
  TORCH_MODULE(ResNet101);
  TORCH_MODULE(ResNet152);
  TORCH_MODULE(ResNext50_32x4d);
  TORCH_MODULE(ResNext101_32x8d);

  // build backbone from json config, we only accept two args:
  // one is the type of backbone and the other is num of frozen stages
  std::shared_ptr<Backbone> build_backbone(const json &opts);
  
} //  end of namespace frcnn

#endif
