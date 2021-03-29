#include <backbones.hpp>


namespace frcnn
{

  void freeze_module(torch::nn::Module *ptr){
    ptr->eval();
    for(auto& para : ptr->parameters()){
      para.requires_grad_(false);
    }
  }

  torch::nn::Conv2d conv3x3(int64_t in,
			    int64_t out,
			    int64_t stride,
			    int64_t groups) {
    torch::nn::Conv2dOptions O(in, out, 3);
    O.padding(1).stride(stride).groups(groups).bias(false);
    return torch::nn::Conv2d(O);
  }
  
  torch::nn::Conv2d conv1x1(int64_t in, int64_t out, int64_t stride) {
    torch::nn::Conv2dOptions O(in, out, 1);
    O.stride(stride).bias(false);
    return torch::nn::Conv2d(O);
  }

  int BasicBlock::expansion = 1;
  int Bottleneck::expansion = 4;

  BasicBlock::BasicBlock(int64_t inplanes,
			 int64_t planes,
			 int64_t stride,
			 torch::nn::Sequential downsample,
			 int64_t groups,
			 int64_t base_width)
    : stride(stride), downsample(downsample) {
    TORCH_CHECK(groups == 1 && base_width == 64,
		"BasicBlock only supports groups=1 and base_width=64");
    
    // Both conv1 and downsample layers downsample the input when stride != 1
    conv1 = conv3x3(inplanes, planes, stride);
    conv2 = conv3x3(planes, planes);
    
    bn1 = torch::nn::BatchNorm2d(planes);
    bn2 = torch::nn::BatchNorm2d(planes);
    
    register_module("conv1", conv1);
    register_module("bn1", bn1);
    register_module("conv2", conv2);
    register_module("bn2", bn2);
    
    if (!downsample.is_empty())
      register_module("downsample", this->downsample);
  }

  Bottleneck::Bottleneck(int64_t inplanes,
			 int64_t planes,
			 int64_t stride,
			 torch::nn::Sequential downsample,
			 int64_t groups,
			 int64_t base_width)
    : stride(stride), downsample(downsample) {
    auto width = int64_t(planes * (base_width / 64.)) * groups;
    
    // Both conv2 and downsample layers downsample the input when stride != 1
    conv1 = conv1x1(inplanes, width);
    conv2 = conv3x3(width, width, stride, groups);
    conv3 = conv1x1(width, planes * expansion);
    
    bn1 = torch::nn::BatchNorm2d(width);
    bn2 = torch::nn::BatchNorm2d(width);
    bn3 = torch::nn::BatchNorm2d(planes * expansion);
    
    register_module("conv1", conv1);
    register_module("bn1", bn1);
    
    register_module("conv2", conv2);
    register_module("bn2", bn2);
    
    register_module("conv3", conv3);
    register_module("bn3", bn3);
    
    if (!downsample.is_empty())
      register_module("downsample", this->downsample);
  }

  torch::Tensor Bottleneck::forward(torch::Tensor X) {
    auto identity = X;
    
    auto out = conv1->forward(X);
    out = bn1->forward(out).relu_();
    
    out = conv2->forward(out);
    out = bn2->forward(out).relu_();
    
    out = conv3->forward(out);
    out = bn3->forward(out);
    
    if (!downsample.is_empty())
      identity = downsample->forward(X);
    
    out += identity;
    return out.relu_();
  }
  
  torch::Tensor BasicBlock::forward(torch::Tensor x) {
    auto identity = x;
    
    auto out = conv1->forward(x);
    out = bn1->forward(out).relu_();
    
    out = conv2->forward(out);
    out = bn2->forward(out);
    
    if (!downsample.is_empty())
      identity = downsample->forward(x);
    
    out += identity;
    return out.relu_();
  }

  template <typename Block>
  void ResNetImpl<Block>::freeze_stages(int stages){
    ASSERT(stages<=4, "freeze stages can not be bigger than 4 for ResNets");
    if (stages <= 0){
      return;
    }
    freeze_module(conv1.get());
    freeze_module(bn1.get());
    auto layers = std::vector<torch::nn::Sequential>({layer1, layer2, layer3, layer3});
    for(int i=0; i<stages; i++){
      freeze_module(layers[i].get());
    }
  }

  template <typename Block>
  void ResNetImpl<Block>::train(bool on){
    torch::nn::Module::train(on);
    freeze_stages(frozen_stages);
    if(on){
      for(auto& module : modules(false)) {
	if (auto M = dynamic_cast<torch::nn::BatchNorm2dImpl*>(module.get())){
	  module->eval();
	}
      }
    }
  }

  ResNet18Impl::ResNet18Impl(int frozen_stages)
    : ResNetImpl({2, 2, 2, 2}, false, 1, 64, frozen_stages) {}
  
  ResNet34Impl::ResNet34Impl(int frozen_stages)
    : ResNetImpl({3, 4, 6, 3}, false, 1, 64, frozen_stages) {}
  
  ResNet50Impl::ResNet50Impl(int frozen_stages)
    : ResNetImpl({3, 4, 6, 3}, false, 1, 64, frozen_stages) {}
  
  ResNet101Impl::ResNet101Impl(int frozen_stages)
    : ResNetImpl({3, 4, 23, 3}, false, 1, 64, frozen_stages) {}
  
  ResNet152Impl::ResNet152Impl(int frozen_stages)
    : ResNetImpl({3, 8, 36, 3}, false, 1, 64, frozen_stages) {}
  
  ResNext50_32x4dImpl::ResNext50_32x4dImpl(int frozen_stages)
    : ResNetImpl({3, 4, 6, 3}, false, 1, 64, frozen_stages) {}
  
  ResNext101_32x8dImpl::ResNext101_32x8dImpl(int frozen_stages)
    : ResNetImpl({3, 4, 23, 3}, false, 1, 64, frozen_stages) {}

  
  // build backbone(resnet series) from "type" keyword in opts
  std::shared_ptr<Backbone> build_backbone(const json &opts){
    std::string type = opts["type"].get<std::string>();
    int frozen = opts["frozen_stages"].get<int>();
    if(type == "resnet18"){
      return std::make_shared<ResNet18Impl>(frozen);
    } else if(type == "resnet34"){
      return std::make_shared<ResNet34Impl>(frozen);
    } else if(type == "resnet50"){
      return std::make_shared<ResNet50Impl>(frozen);
    } else if(type == "resnet101"){
      return std::make_shared<ResNet101Impl>(frozen);
    } else if(type == "resnet152"){
      return std::make_shared<ResNet152Impl>(frozen);
    } else if(type == "resnext50"){
      return std::make_shared<ResNext50_32x4dImpl>(frozen);
    } else if(type == "resnext101"){
      return std::make_shared<ResNext101_32x8dImpl>(frozen);
    } else {
      throw std::runtime_error("unsupported backbone type: " + type);
    }
  }

  
} //namespace frcnn
