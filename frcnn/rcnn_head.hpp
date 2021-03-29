#ifndef RCNN_HEAD_HPP
#define RCNN_HEAD_HPP

#include <torch/torch.h>
#include <cvops.h>
#include <json.hpp>
#include <utils.hpp>
#include <bbox.hpp>
#include <data.hpp>
#include <losses.hpp>

namespace frcnn
{
  using json = nlohmann::json;
  namespace F = torch::nn::functional;
  using namespace torch::indexing;

  /**
     Due to the use of FPN, the features are in multi-levels.
     It first maps rois to different feature levels by roi sizes, 
     then do RoIPool or RoIAlign on different levels.
   */
  class RoIExtractorImpl : public torch::nn::Module
  {
  public:
    enum class roi_type {RoIAlign, RoIPool};
    RoIExtractorImpl(int out_channels,
		     const std::vector<int> &featmap_strides, // size must be smaller than number of features recieved
		     const std::string &type,  // support either RoIAlign or RoIPool
		     const std::vector<int> &output_size, // [h, w], e.g. [7, 7]
		     int sampling_ratio=0, // only used in RoIAlign
		     int finest_scale=56);
    RoIExtractorImpl(const json &opts);
    
    // return fix-sized roi pooling result
    torch::Tensor forward(std::vector<torch::Tensor> feats,
			  torch::Tensor rois);
    // map rois to different feature levels by sizes of rois.
    torch::Tensor map_roi_levels(torch::Tensor rois, int num_levels);
  private:
    int _out_channels;
    std::vector<int> _featmap_strides;
    std::string _type;
    std::vector<int> _output_size;
    int _sampling_ratio;
    int _finest_scale;
    roi_type _roi_type{roi_type::RoIAlign};
  };
  TORCH_MODULE(RoIExtractor);
  
  /**
     Typically RCNN consists of two fc layers followed by two parallel fc layers for classification and regression.
   */
  class RCNNHeadImpl : public torch::nn::Module
  {
  public:
    RCNNHeadImpl(int in_channels,
		 const std::vector<int> &fc_out_channels,
		 int num_classes,
		 int roi_feat_size,
		 const json &roi_extractor_opts,
		 const json &bbox_coder_opts,
		 const json &loss_cls_opts,
		 const json &loss_bbox_opts,
		 const json &train_opts,
		 const json &test_opts);
    
    RCNNHeadImpl(const json &opts);

    // return cls_outs and bbox_outs
    std::tuple<torch::Tensor, torch::Tensor> forward
    (std::vector<torch::Tensor> feats, torch::Tensor rois);
    // return loss_cls and loss_bbox
    std::tuple<torch::Tensor, torch::Tensor> forward_train(std::vector<torch::Tensor> feats,
							   torch::Tensor rois,
							   ImgData &img_data);
    // return det results
    std::tuple<torch::Tensor, torch::Tensor, torch::Tensor> forward_test
    (std::vector<torch::Tensor> feats, torch::Tensor rois, ImgData &img_data);
  private:
    int _in_channels;
    std::vector<int> _fc_out_channels;
    int _num_classes;
    int _roi_feat_size;
    json _roi_extractor_opts;
    json _bbox_coder_opts;
    json _train_opts;
    json _test_opts;
    json _loss_cls_opts;
    json _loss_bbox_opts;

    BBoxRegressCoder _bbox_coder;
    BBoxAssigner _bbox_assigner;
    
    RoIExtractor _roi_extractor{nullptr};
    torch::nn::ModuleList _shared_fcs{nullptr};
    torch::nn::Linear _classifier{nullptr};
    torch::nn::Linear _regressor{nullptr};
    std::shared_ptr<Loss> _loss_cls{nullptr};
    std::shared_ptr<Loss> _loss_bbox{nullptr};
      
  };
  TORCH_MODULE(RCNNHead);

}



#endif
