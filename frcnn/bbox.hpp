#ifndef BBOX_HPP
#define BBOX_HPP
#include <torch/torch.h>
#include <utils.hpp>
#include <cvops.h>
#include <json.hpp>

namespace frcnn
{
  using json = nlohmann::json;
  /**
     convert btw bbox and delta given a base_bbox
   */
  class BBoxRegressCoder
  {
  public:
    BBoxRegressCoder(const std::vector<float> &means={0.0, 0.0, 0.0, 0.0},
		     const std::vector<float> &stds ={1.0, 1.0, 1.0, 1.0});
    BBoxRegressCoder(const json &opts);
    // calculate delta given base bboxes and target bboxes
    torch::Tensor encode(torch::Tensor base, torch::Tensor bboxes);
    // calculate bboxes given base bboxes and delta
    torch::Tensor decode(torch::Tensor base, torch::Tensor delta,
			 const std::vector<int64_t> &max_shape = std::vector<int64_t>());
  private:
    std::vector<float> _means;
    std::vector<float> _stds;
  };

  // calculate IoUs between a and b, return an IoU table of size [a.size(0), b.size(0)]
  torch::Tensor calc_iou(torch::Tensor a, torch::Tensor b);
  // element-wise IoU, assume a.size(0)==b.size(0)
  torch::Tensor elem_iou(torch::Tensor a, torch::Tensor b);
  // giou 
  torch::Tensor giou(torch::Tensor a, torch::Tensor b);


  // contain results of an assignment, note that bboxes may contain gt bboxes
  struct AssignResult
  {
    torch::Tensor bboxes;     // rois
    // -2: ignore, -1: negative, >=0: indices(not gt labels) of gt_bboxes in current image
    torch::Tensor assigned_inds; 
    torch::Tensor gt_bboxes;
    torch::Tensor gt_labels;
    torch::Tensor is_gt; // 1: is gt, 0: not gt
  };

  // assign gt_bboxes to rois, followed by random sampling
  class BBoxAssigner
  {
  public:
    BBoxAssigner(float pos_iou_thr,  // above which will be considered positive
		 float neg_iou_thr,  // below which will be considered negative
		 float min_pos_iou,  // below which will never be considered positive
		 int samp_num,       // total samples to sample
		 float pos_frac,     // fraction of positive samples
		 bool add_gt);       // if add gt to proposal bboxes
    BBoxAssigner(const json &opts);
    
    // assign gt_bboxes to bboxes and sample the result
    AssignResult assign(torch::Tensor bboxes, torch::Tensor gt_bboxes, torch::Tensor gt_labels);
  private:
    float _pos_iou_thr;
    float _neg_iou_thr;
    float _min_pos_iou;
    int _samp_num;
    float _pos_frac;
    bool _add_gt;
  };

  // do nms on bboxes with different labels at the same time
  // the trick is to move bboxes of different label far away from each other
  // return keep inds ordered by score
  torch::Tensor batched_nms(torch::Tensor bboxes, // bboxes to apply NMS to
			    torch::Tensor scores, // scores
			    torch::Tensor labels, // labels 
			    float iou_thr);
  
  // do nms on all bboxes of the same category, and repeat this for all categories
  // under rare cases one bbox may have multiple labels
  std::tuple<torch::Tensor, torch::Tensor, torch::Tensor>
  multiclass_nms(torch::Tensor bboxes,  // [n, #class*4] or [n, 4]
		 torch::Tensor scores,  // [n, #class]
		 float iou_thr,
		 float score_thr=-1,
		 int max_num=-1);
  
}

#endif
