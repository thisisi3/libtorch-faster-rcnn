#ifndef RPN_HEAD_HPP
#define RPN_HEAD_HPP

#include <torch/torch.h>
#include <bbox.hpp>
#include <anchor.hpp>
#include <data.hpp>
#include <losses.hpp>

namespace frcnn
{
  namespace F = torch::nn::functional;
  /*
    Implementation of RPN.
   */
  class RPNHeadImpl : public torch::nn::Module
  {
  public:
    RPNHeadImpl(int in_channels,
		int feat_channels,
		const json &anchor_opts,
		const json &bbox_coder_opts,
		const json &loss_cls_opts,
		const json &loss_bbox_opts,
		const json &train_opts,
		const json &test_opts);
    RPNHeadImpl(const json &opts);
    
    std::tuple<std::vector<torch::Tensor>, std::vector<torch::Tensor>>
    forward(std::vector<torch::Tensor> feats);

    // return cls_loss, bbox_loss and proposals
    std::tuple<torch::Tensor, torch::Tensor, torch::Tensor> forward_train
    (std::vector<torch::Tensor> feats, ImgData &img_data); 

    // return proposals
    torch::Tensor forward_test
    (std::vector<torch::Tensor> feats, ImgData &img_data);

    // get_proposal
    torch::Tensor get_proposals
    (std::vector<torch::Tensor> anchor_list,
     std::vector<torch::Tensor> cls_out_list,
     std::vector<torch::Tensor> bbox_out_list,
     const std::vector<int64_t> &img_shape,
     const json &opts);
  private:
    int _in_channels;
    int _feat_channels;
    json _anchor_opts;
    json _bbox_coder_opts;
    json _train_opts;
    json _test_opts;
    json _loss_cls_opts;
    json _loss_bbox_opts;
    
    BBoxRegressCoder _bbox_coder;
    AnchorGenerator _anchor_generator;
    BBoxAssigner _bbox_assigner;
    
    int _class_channels{1};
    
    torch::nn::Conv2d _conv{nullptr};
    torch::nn::Conv2d _classifier{nullptr};
    torch::nn::Conv2d _regressor{nullptr};
    std::shared_ptr<Loss> _loss_cls{nullptr};
    std::shared_ptr<Loss> _loss_bbox{nullptr};
    
  };
  TORCH_MODULE(RPNHead);
  

}

#endif
