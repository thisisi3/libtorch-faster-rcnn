#include <anchor.hpp>

namespace frcnn
{
  /*
    class AnchorGenerator
   */
  AnchorGeneratorImpl::AnchorGeneratorImpl(const json &opts)
    : AnchorGeneratorImpl(opts["anchor_scales"].get<std::vector<float>>(),
			  opts["anchor_strides"].get<std::vector<float>>(),
			  opts["anchor_ratios"].get<std::vector<float>>(),
			  opts["center_offset"].get<float>())
  { /* build AnchorGenerator by json-format options */}
  
  AnchorGeneratorImpl::AnchorGeneratorImpl(const std::vector<float> &anchor_scales,
					   const std::vector<float> &anchor_strides,
					   const std::vector<float> &anchor_ratios,
					   float center_offset)
    : _anchor_scales(anchor_scales),
      _anchor_strides(anchor_strides),
      _anchor_ratios(anchor_ratios),
      _center_offset(center_offset)
  {
    ASSERT(center_offset <= 1.0 && center_offset >=0, "center offset must be in [0.0, 1.0]");
    auto scales = torch::tensor(anchor_scales, torch::kFloat32),
      ratios = torch::tensor(anchor_ratios, torch::kFloat32),
      strides = torch::tensor(anchor_strides, torch::kFloat32);
    // [num_strides, num_scales, 1]
    scales = (strides.view({-1, 1}) * scales).unsqueeze(-1); 
    // [num_strides, num_scales, num_ratios]
    auto w = scales / ratios.sqrt(), h = scales * ratios.sqrt(); 
    // [num_strides, num_scales, num_ratios, 4], center at upper-left corner
    _base_anchors = torch::stack({-w/2, -h/2, w/2, h/2}, -1); 
    _base_anchors += strides.view({-1, 1, 1, 1}) * _center_offset; // move center by offset
    register_buffer("base_anchors", _base_anchors); // device follows module's device
  }

  std::vector<torch::Tensor> AnchorGeneratorImpl::get_anchors
  (const std::vector<std::vector<int64_t>> grid_sizes){
    std::vector<torch::Tensor> anchors;
    for(int i=0; i<grid_sizes.size(); i++){
      auto grid_size = grid_sizes[i];
      auto stride = _anchor_strides[i];
      auto base_anchor = _base_anchors[i]; // [num_scales, num_ratios, 4]
      auto grid_x = grid_size[1], grid_y = grid_size[0];
      auto x_offset = torch::arange(grid_x, base_anchor.device()) * stride; // [grid_x]
      auto y_offset = torch::arange(grid_y, base_anchor.device()) * stride; // [grid_y]
      
      auto anchor = base_anchor.repeat({grid_y, grid_x, 1, 1, 1}); // [grid_y, grid_x, num_scales, num_ratios, 4]
      auto shape = anchor.sizes().vec();
      // from [grid_y, grid_x, num_scales, num_ratios, 4] 
      // to   [grid_y, grid_x, num_scales, num_ratios, 2, 2]
      // so that we can add offset to the last dim
      shape.pop_back(); shape.push_back(2); shape.push_back(2); // can't find an easier way
      anchor = anchor.view(shape);
      // move x by x_offset and y by y_offset
      anchor.index_put_({"...", 1}, y_offset.view({-1, 1, 1, 1, 1})+anchor.index({"...", 1}));
      anchor.index_put_({"...", 0}, x_offset.view({-1, 1, 1, 1})+anchor.index({"...", 0}));
      // from [grid_y, grid_x, num_scales, num_ratios, 2, 2] 
      // to   [grid_y, grid_x, num_scales, num_ratios, 4] 
      shape.pop_back(); shape.pop_back(); shape.push_back(4);
      anchor = anchor.view(shape);
      // the shape of anchor for current stride is [grid_y, grid_x, num_scales, num_ratios, 4]
      anchors.push_back(anchor);
    }
    return anchors;
  }

  int AnchorGeneratorImpl::num_anchors(){
    return _anchor_scales.size() * _anchor_ratios.size();
  }

  
}
