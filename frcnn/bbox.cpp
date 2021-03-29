#include <bbox.hpp>

namespace frcnn
{
  /*
    class BBoxRegressCoder
   */
  BBoxRegressCoder::BBoxRegressCoder(const std::vector<float> &means,
				     const std::vector<float> &stds)
    : _means(means), _stds(stds)
  {
    ASSERT(means.size()==4 && stds.size()==4, "target means and stds must have size 4");
  }

  BBoxRegressCoder::BBoxRegressCoder(const json &opts)
    : _means(opts["means"].get<std::vector<float>>()),
      _stds(opts["stds"].get<std::vector<float>>())
  { }

  torch::Tensor BBoxRegressCoder::encode(torch::Tensor base, torch::Tensor bboxes){
    base = base.to(torch::kFloat32);
    bboxes = bboxes.to(torch::kFloat32);
    auto base_xywh = xyxy2xywh(base);
    auto bboxes_xywh = xyxy2xywh(bboxes);
    auto base_wh = base_xywh.index({Slice(), Slice(2, None)});
    auto xy_delta = bboxes_xywh.index({Slice(), Slice(None, 2)}) - base_xywh.index({Slice(), Slice(None, 2)});
    xy_delta = xy_delta / base_wh;
    auto wh_delta = torch::log
      (bboxes_xywh.index({Slice(), Slice(2, None)}) / base_wh);
    auto delta = torch::cat({xy_delta, wh_delta}, 1);
    delta = delta - torch::tensor(_means).to(base).view({1, 4});
    delta = delta / torch::tensor(_stds).to(base).view({1, 4});
    return delta;
  }

  torch::Tensor BBoxRegressCoder::decode(torch::Tensor base, torch::Tensor delta,
					 const std::vector<int64_t> &max_shape){
    base = base.to(torch::kFloat32);
    delta = delta.to(torch::kFloat32);
    delta = delta * torch::tensor(_stds).to(base).view({1, 4});
    delta = delta + torch::tensor(_means).to(base).view({1, 4});
    auto base_xywh = xyxy2xywh(base);
    auto base_wh = base_xywh.index({Slice(), Slice(2, None)});
    auto bboxes_xy = delta.index({Slice(), Slice(None, 2)}) * base_wh
      + base_xywh.index({Slice(), Slice(None, 2)});
    auto bboxes_wh = torch::exp(delta.index({Slice(), Slice(2, None)})) * base_wh;
    auto bboxes =  torch::cat({bboxes_xy, bboxes_wh}, 1);
    bboxes = xywh2xyxy(bboxes);
    if (max_shape.size() >= 2){
      bboxes = restrict_bbox(bboxes, max_shape);
    }
    return bboxes;
  }


  /**
     class BBoxAssigner
   */

  BBoxAssigner::BBoxAssigner(float pos_iou_thr,
			     float neg_iou_thr,
			     float min_pos_iou,
			     int samp_num,
			     float pos_frac,
			     bool add_gt)
    : _pos_iou_thr(pos_iou_thr),
      _neg_iou_thr(neg_iou_thr),
      _min_pos_iou(min_pos_iou),
      _samp_num(samp_num),
      _pos_frac(pos_frac),
      _add_gt(add_gt)
  {
    ASSERT(pos_iou_thr>=0 && pos_iou_thr<=1.0, "pos_iou_thr must be in [0.0, 1.0]");
    ASSERT(neg_iou_thr>=0 && neg_iou_thr<=1.0, "neg_iou_thr must be in [0.0, 1.0]");
    ASSERT(min_pos_iou>=0 && min_pos_iou<=1.0, "min_pos_iou must be in [0.0, 1.0]");
    ASSERT(pos_iou_thr>=neg_iou_thr, "pos_iou_thr must not be less than neg_iou_thr");
    ASSERT(pos_frac>=0 && pos_frac<=1.0, "pos_frac must be in [0.0, 1.0]");
  }

  BBoxAssigner::BBoxAssigner(const json &opts)
    : BBoxAssigner(opts["pos_iou_thr"].get<float>(),
		   opts["neg_iou_thr"].get<float>(),
		   opts["min_pos_iou"].get<float>(),
		   opts["samp_num"].get<int>(),
		   opts["pos_frac"].get<float>(),
		   opts["add_gt"].get<bool>())
  { }

  AssignResult BBoxAssigner::assign(torch::Tensor bboxes,
				    torch::Tensor gt_bboxes,
				    torch::Tensor gt_labels){
    AssignResult assign_result;
    assign_result.gt_bboxes = gt_bboxes;
    assign_result.gt_labels = gt_labels;
    int num_gts = gt_bboxes.size(0);
    if(_add_gt){ bboxes = torch::cat({gt_bboxes, bboxes}, 0); }
    int num_bboxes = bboxes.size(0);
    auto assigned_inds = torch::full // first assign -2(ignore) everywhere
      ({num_bboxes}, -2, torch::TensorOptions().dtype(torch::kInt64).device(bboxes.device()));
    // [n, m], n=num of bboxes and m=num of gt
    auto iou_tab = calc_iou(bboxes, gt_bboxes); 
    auto max_gt_res = iou_tab.max(1);
    auto max_gt_iou = std::get<0>(max_gt_res), max_gt_idx = std::get<1>(max_gt_res);
    assigned_inds.index_put_({max_gt_iou <  _neg_iou_thr}, -1); // find negative inds
    assigned_inds.index_put_({max_gt_iou >= _pos_iou_thr},  1); // find positive inds
    
    auto max_bbox_res = iou_tab.max(0);
    auto max_bbox_iou = std::get<0>(max_bbox_res);
    // dimensions: ([n, m] == m) & (m >= 1)
    auto equal_max_places = torch::logical_and((iou_tab == max_bbox_iou),(max_bbox_iou >= _min_pos_iou));
    auto equal_max_idx = std::get<1>(equal_max_places.max(1)); // a trick to get gt idx 
    equal_max_places = (equal_max_places.sum(1) > 0); // a trick to find 1D equal_max_places
    // next update assigned_inds and max_gt_idx
    assigned_inds.index_put_({equal_max_places}, 1);
    max_gt_idx.index_put_({equal_max_places}, equal_max_idx.index(equal_max_places));
    //
    // next sample assigned result
    //
    int pos_expected = (int)(_samp_num * _pos_frac);
    auto pos_inds = (assigned_inds == 1).nonzero();
    int pos_sampled = pos_inds.size(0);
    if (pos_expected < pos_inds.size(0)){
      auto oppo_select = rand_choice(pos_inds, (pos_inds.size(0) - pos_expected));
      assigned_inds.index_put_({oppo_select}, -2);
      pos_sampled = pos_expected;
    }

    int neg_expected = _samp_num - pos_sampled;
    auto neg_inds = (assigned_inds == -1).nonzero();
    int neg_sampled = neg_inds.size(0);
    if (neg_expected < neg_inds.size(0)){
      auto oppo_select = rand_choice(neg_inds, (neg_inds.size(0) - neg_expected));
      assigned_inds.index_put_({oppo_select}, -2);
      neg_sampled = neg_expected;
    }
    // update positive inds with gt idx
    pos_inds = (assigned_inds == 1);
    assigned_inds.index_put_({pos_inds}, max_gt_idx.index({pos_inds}));
    pos_inds.index_put_({Slice(num_gts, None)}, false); // find is_gt mask
    // fill in AssignResult
    assign_result.bboxes = bboxes;
    assign_result.assigned_inds = assigned_inds;
    assign_result.is_gt = pos_inds;
    
    return assign_result;
    
  }

  torch::Tensor calc_iou(torch::Tensor a, torch::Tensor b){
    a = a.to(torch::kFloat32);
    b = b.to(torch::kFloat32);
    ASSERT(a.dim()==2 && b.dim()==2 && a.size(1)==4 && b.size(1)==4,
	   "in order to calculate IoU, tensors must have [n, 4] dimensions");
    a = a.view({-1, 1, 4});
    // top-left
    auto tl = torch::max(a.index({Slice(), Slice(), Slice(None, 2)}),
			 b.index({Slice(), Slice(None, 2)}));
    // bottom-right
    auto br = torch::min(a.index({Slice(), Slice(), Slice(2, None)}),
			 b.index({Slice(), Slice(2, None)}));
    auto inter_wh = br - tl;
    auto pos_mask = torch::logical_and
      ((inter_wh.index({Slice(), Slice(), 0}) > 0) ,(inter_wh.index({Slice(), Slice(), 1}) > 0));
    auto inter_area =
      inter_wh.index({Slice(), Slice(), 0}) *
      inter_wh.index({Slice(), Slice(), 1});
    inter_area.index_put_({torch::logical_not(pos_mask)}, 0.0);
    auto a_area = bbox_area(a.view({-1, 4})).view({-1, 1});
    auto b_area = bbox_area(b);
    auto iou = inter_area / (a_area + b_area - inter_area);
    return iou;
  }

  // use calc_iou, which is sub-optimal
  torch::Tensor elem_iou(torch::Tensor a, torch::Tensor b){
    ASSERT(a.sizes()==b.sizes() && a.dim()==2 && b.dim()==2 && a.size(1)==4 && b.size(1)==4,
	   "in order to calculate element-wise iou, bboxes must have same and valid dimensions");
    auto ious = calc_iou(a, b);
    return ious.diag();
  }

  torch::Tensor giou(torch::Tensor a, torch::Tensor b){
    double eps = 1e-6;
    a = a.to(torch::kFloat32);
    b = b.to(torch::kFloat32);
    auto tl = torch::max(a.index({Slice(), Slice(None, 2)}), b.index({Slice(), Slice(None, 2)}));
    auto br = torch::min(a.index({Slice(), Slice(2, None)}), b.index({Slice(), Slice(2, None)}));
    auto inter_wh = br - tl;
    auto pos_mask = torch::logical_and(inter_wh.index({Slice(), 0})>0, inter_wh.index({Slice(), 1})>0);
    auto inter_area = inter_wh.index({Slice(), 0}) * inter_wh.index({Slice(), 1});
    inter_area.index_put_({torch::logical_not(pos_mask)}, 0.0);
    auto a_area = bbox_area(a);
    auto b_area = bbox_area(b);
    auto union_area = a_area + b_area - inter_area;
    auto iou = inter_area / (union_area + eps);
    auto rect_min_xy = torch::min(a.index({Slice(), Slice(None, 2)}), b.index({Slice(), Slice(None, 2)}));
    auto rect_max_xy = torch::max(a.index({Slice(), Slice(2, None)}), b.index({Slice(), Slice(2, None)}));
    auto rect = torch::cat({rect_min_xy, rect_max_xy}, 1);
    auto rect_area = bbox_area(rect);
    return iou - (rect_area - union_area)/(rect_area + eps);
  }

  torch::Tensor batched_nms(torch::Tensor bboxes, // bboxes to apply NMS to
                            torch::Tensor scores, // scores
                            torch::Tensor labels, // labels
                            float iou_thr){
    if (bboxes.size(0)==0){
      return torch::empty({0}).to(torch::kLong);
    }
    auto nms_bboxes = bboxes;
    if (labels.defined() && labels.numel()==bboxes.size(0)){
      auto max_range = bboxes.max();
      nms_bboxes = bboxes + (labels * max_range).to(bboxes).view({bboxes.size(0), 1});
    }
    return nms(nms_bboxes, scores, iou_thr);
  }

  std::tuple<torch::Tensor, torch::Tensor, torch::Tensor> 
  multiclass_nms(torch::Tensor bboxes,  // [n, #class*4] or [n, 4]                                      
		 torch::Tensor scores,  // [n, #class]                                                  
		 float iou_thr,
		 float score_thr,
		 int max_num){
    int num_classes = scores.size(1), num_bboxes=bboxes.size(0);
    if (bboxes.size(1)>4){
      bboxes = bboxes.view({-1, 4});
    } else {
      bboxes = bboxes.view({num_bboxes, 1, 4});
      bboxes = bboxes.repeat({1, num_classes, 1}).view({-1, 4});
    }
    scores = scores.reshape({-1});
    auto labels = torch::arange(num_classes).to(scores).view({1, -1}).repeat({num_bboxes, 1}).view({-1});
    auto chosen = (scores>=score_thr);
    bboxes = bboxes.index({chosen});
    scores = scores.index({chosen});
    labels = labels.index({chosen});
    auto keep = batched_nms(bboxes, scores, labels, iou_thr);
    if (max_num != -1 & keep.size(0) > max_num){
      keep = keep.index({Slice(0, max_num)});
    }
    return std::make_tuple(bboxes.index({keep}), scores.index({keep}), labels.index({keep}));

  }


  
}
