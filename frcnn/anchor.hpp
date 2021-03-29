#ifndef ANCHOR_HPP
#define ANCHOR_HPP

#include <torch/torch.h>
#include <json.hpp>
#include <utils.hpp>

namespace frcnn
{
  using json = nlohmann::json;
  using namespace torch::indexing;
  
  class AnchorGeneratorImpl : public torch::nn::Module
  {
  public:
    // Center_offset used to be 0.5, i.e. center of base anchors is at the center of grid.
    // Later people tend to let center_offset=0, i.e. base anchors are centered at upper-left of the grid.
    AnchorGeneratorImpl(const std::vector<float> &anchor_scales={8},
			const std::vector<float> &anchor_strides={4, 8, 16, 32, 64},
			const std::vector<float> &anchor_ratios={0.5, 1.0, 2.0},
			float center_offset=0.0);

    AnchorGeneratorImpl(const json &opts);
    /**
       Need two steps to generate anchors: 
         1, generate base_anchors. 
	 2, tile base_anchors along x-axis and y-axis.
       Generate base_anchors:
         1, scale = stride * scale
         2, w*h = scale^2 
	 3, h/w = ratio    
	 4, center bboxes at stride*center_offset
     */
    std::vector<torch::Tensor> get_anchors(const std::vector<std::vector<int64_t>> grid_sizes);
    // number of base anchors
    int num_anchors();
  private:
    std::vector<float> _anchor_scales;
    // decides the strides and impacts the scales of anchors in current feat level
    std::vector<float> _anchor_strides;  
    std::vector<float> _anchor_ratios;   
    float _center_offset;
    torch::Tensor _base_anchors;

  }; // class AnchorGeneratorImpl
  TORCH_MODULE(AnchorGenerator);

}

#endif
