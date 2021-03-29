#ifndef DETECTORS_HPP
#define DETECTORS_HPP

#include <backbones.hpp>
#include <necks.hpp>
#include <rpn_head.hpp>
#include <rcnn_head.hpp>

namespace frcnn
{
  /**
     Implementation of FasterRCNN
   */
  class FasterRCNNImpl : public torch::nn::Module
  {
  public:
    FasterRCNNImpl(const json &backbone_opts,
		   const json &fpn_opts,
		   const json &rpn_opts,
		   const json &rcnn_opts);
    // return a map of losses
    std::map<std::string, torch::Tensor> forward_train
    (torch::Tensor img_tsr, ImgData &img_data);
    // return det_bboxes, det_scores, det_labels
    std::tuple<torch::Tensor, torch::Tensor, torch::Tensor> forward_test
    (torch::Tensor img_tsr, ImgData &img_data);
  private:
    json _backbone_opts;
    json _fpn_opts;
    json _rpn_opts;
    json _rcnn_opts;

    std::shared_ptr<Backbone> _backbone{nullptr};
    FPN _neck{nullptr};
    RPNHead _rpn_head{nullptr};
    RCNNHead _rcnn_head{nullptr};
  };
  TORCH_MODULE(FasterRCNN);
  
}

#endif
