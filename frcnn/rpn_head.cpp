#include <rpn_head.hpp>

namespace frcnn
{
  RPNHeadImpl::RPNHeadImpl(int in_channels,
			   int feat_channels,
			   const json &anchor_opts,
			   const json &bbox_coder_opts,
			   const json &loss_cls_opts,
			   const json &loss_bbox_opts,
			   const json &train_opts,
			   const json &test_opts)
    : _in_channels(in_channels),
      _feat_channels(feat_channels),
      _anchor_opts(anchor_opts),
      _bbox_coder_opts(bbox_coder_opts),
      _loss_cls_opts(loss_cls_opts),
      _loss_bbox_opts(loss_bbox_opts),
      _train_opts(train_opts),
      _test_opts(test_opts),
      _bbox_coder(bbox_coder_opts),
      _anchor_generator(anchor_opts),
      _bbox_assigner(train_opts["bbox_assigner_opts"].get<json>())
  {
    _class_channels = 1; // use sigmoid, so class channel is 1
    _conv = torch::nn::Conv2d(torch::nn::Conv2dOptions(in_channels, feat_channels, 3)
			     .stride(1).padding(1));
    _classifier = torch::nn::Conv2d
      (torch::nn::Conv2dOptions(feat_channels,
				_anchor_generator->num_anchors()*_class_channels, 1));
    _regressor = torch::nn::Conv2d
      (torch::nn::Conv2dOptions(feat_channels,
				_anchor_generator->num_anchors()*4, 1));
    _loss_cls  = build_loss(loss_cls_opts);
    _loss_bbox = build_loss(loss_bbox_opts);
    register_module("conv", _conv);
    register_module("classifier", _classifier);
    register_module("regressor", _regressor);
    register_module("anchor_generator", _anchor_generator);
    register_module("loss_cls", _loss_cls);
    register_module("loss_bbox", _loss_bbox);

    // init_weights
    torch::nn::init::normal_(_conv->weight,       0, 0.01);
    torch::nn::init::normal_(_classifier->weight, 0, 0.01);
    torch::nn::init::normal_(_regressor->weight,  0, 0.01);
    torch::nn::init::constant_(_conv->bias,       0);
    torch::nn::init::constant_(_classifier->bias, 0);
    torch::nn::init::constant_(_regressor->bias,  0);
  }

  RPNHeadImpl::RPNHeadImpl(const json &opts)
    : RPNHeadImpl(opts["in_channels"].get<int>(),
		  opts["feat_channels"].get<int>(),
		  opts["anchor_opts"].get<json>(),
		  opts["bbox_coder_opts"].get<json>(),
		  opts["loss_cls_opts"].get<json>(),
		  opts["loss_bbox_opts"].get<json>(),
		  opts["train_opts"].get<json>(),
		  opts["test_opts"].get<json>())
  {/* construct RPNHead by json-format options */ }

  std::tuple<std::vector<torch::Tensor>, std::vector<torch::Tensor>>
  RPNHeadImpl::forward(std::vector<torch::Tensor> feats){
    std::vector<torch::Tensor> cls_outs, bbox_outs; // cls/bbox outputs for all feature levels
    for(auto &x : feats){
      x = _conv->forward(x).relu_();
      cls_outs.push_back(_classifier->forward(x));
      bbox_outs.push_back(_regressor->forward(x));
    }
    return std::make_tuple(cls_outs, bbox_outs);
  }

  // return cls_loss, bbox_loss and proposals
  std::tuple<torch::Tensor, torch::Tensor, torch::Tensor>
  RPNHeadImpl::forward_train(std::vector<torch::Tensor> feats, ImgData &img_data){
    auto rpn_outs = forward(feats);
    // cls_out  has shape: [1, num_anchors*classs_channels, grid_h, grid_w]
    auto cls_out_list = std::get<0>(rpn_outs);
    // bbox_out has shape: [1, num_anchors*4, grid_h, grid_w]
    auto bbox_out_list = std::get<1>(rpn_outs);
    auto grid_sizes = get_grid_size(feats);
    // anchors has shape:  [grid_h, grid_w, num_anchors, 4]
    auto anchor_list = _anchor_generator->get_anchors(grid_sizes);
    // next unify the shapes
    cls_out_list = batch_permute(cls_out_list, {0, 2, 3, 1});
    bbox_out_list = batch_permute(bbox_out_list, {0, 2, 3, 1});
    cls_out_list = batch_reshape(cls_out_list, {-1, _class_channels});
    bbox_out_list = batch_reshape(bbox_out_list, {-1, 4});
    anchor_list = batch_reshape(anchor_list, {-1, 4});
    
    /*************** get proposals *************/
    auto proposals = get_proposals(anchor_list, cls_out_list, bbox_out_list,
				   img_data.img_shape, _train_opts);
    /*************** get proposals *************/
    auto cls_outs = torch::cat(cls_out_list, 0);
    auto bbox_outs = torch::cat(bbox_out_list, 0);
    auto anchors = torch::cat(anchor_list, 0);

    // get general assigned results
    auto assigned_result = _bbox_assigner.assign(anchors, img_data.gt_bboxes, img_data.gt_labels);
    auto &assigned_inds = assigned_result.assigned_inds;
    auto chosen = (assigned_inds >= -1);
    auto tar_inds = assigned_inds.index({chosen});
    auto pos_mask = (tar_inds >= 0);

    // next calc cls loss
    auto pred_cls = cls_outs.index({chosen}).view(-1);
    auto tar_labels = pos_mask.to(torch::kInt64); // special case for RPN
    auto loss_cls = _loss_cls->forward(pred_cls, tar_labels.to(torch::kFloat32), tar_inds.size(0));

    // next calc reg loss
    auto tar_anchors = assigned_result.bboxes.index({chosen});
    // negative bboxes are assigned with the last gt(-1), they will be ignored anyway
    auto tar_bboxes = assigned_result.gt_bboxes.index({tar_inds}); 

    auto bbox_pred = torch::Tensor(), bbox_tar = torch::Tensor();
    if (_loss_bbox_opts["type"].get<std::string>() == "GIoULoss"){
      bbox_tar = tar_bboxes;
      bbox_pred = _bbox_coder.decode(tar_anchors, bbox_outs.index({chosen}));
    } else {
      bbox_tar = _bbox_coder.encode(tar_anchors, tar_bboxes);
      bbox_pred = bbox_outs.index({chosen});
    }
    auto loss_bbox = _loss_bbox->forward(bbox_pred.index({pos_mask}), bbox_tar.index({pos_mask}), tar_inds.size(0));
    
    return std::make_tuple(loss_cls, loss_bbox, proposals);
  }

  /*
   *  method get_proposals()
   *
   */
  // assume opts defines
  //   nms_pre: number of proposals to select for each feature level
  //   nms_post: number of proposals to select after batched nms
  //   nms_thr: iou_thr to be used in NMS
  //   min_bbox_size: filter bbox with size < min_bbox_size
  torch::Tensor RPNHeadImpl::get_proposals
  (std::vector<torch::Tensor> anchor_list,    // [n, 4]
   std::vector<torch::Tensor> cls_out_list,   // [n, 1]
   std::vector<torch::Tensor> bbox_out_list,  // [n, 4]
   const std::vector<int64_t> &img_shape,     // (h, w)
   const json &opts)
  {
    ASSERT(cls_out_list.size() == bbox_out_list.size(),
	   "cls_outs and bbox_outs must have same number of levels");

    std::vector<torch::Tensor> lvls_score, lvls_delta, lvls_anchor, lvls_idx;
    for(int i=0; i<cls_out_list.size(); i++){
      auto cls_out = cls_out_list[i], bbox_out = bbox_out_list[i], anchor = anchor_list[i];
      auto cls_score = cls_out.view(-1).sigmoid();
      int nms_pre = -1;
      if (opts.find("nms_pre")!=opts.end() && (nms_pre = opts["nms_pre"])<cls_score.size(0)){
	auto topk_inds = std::get<1>(cls_score.topk(nms_pre));
	cls_score = cls_score.index({topk_inds});
	bbox_out = bbox_out.index({topk_inds});
	anchor = anchor.index({topk_inds});
      }
      lvls_score.push_back(cls_score);
      lvls_delta.push_back(bbox_out);
      lvls_anchor.push_back(anchor);
      lvls_idx.push_back(torch::full({cls_score.size(0)}, i,
				     torch::kInt64).to(cls_score.device()));
    }

    auto all_score = torch::cat(lvls_score, 0);
    auto all_delta = torch::cat(lvls_delta, 0);
    auto all_anchor = torch::cat(lvls_anchor, 0);
    auto all_idx = torch::cat(lvls_idx, 0);
    auto all_bbox = _bbox_coder.decode(all_anchor, all_delta, img_shape);
    
    int min_bbox_size = 0;
    if(opts.find("min_bbox_size")!=opts.end() && (min_bbox_size=opts["min_bbox_size"]) > 0){
      auto large_mask =
	((all_bbox.index({Slice(), 2}) - all_bbox.index({Slice(), 0})) >= min_bbox_size) &
	((all_bbox.index({Slice(), 3}) - all_bbox.index({Slice(), 1})) >= min_bbox_size);
      all_score = all_score.index({large_mask});
      all_bbox = all_bbox.index({large_mask});
      all_idx = all_idx.index({large_mask});
    }
    
    auto keep = batched_nms(all_bbox, all_score, all_idx, opts["nms_thr"].get<float>());
    // proposals are [n, 5] tensors where last index is score
    auto proposals
      = torch::cat({all_bbox.index({keep}), all_score.index({keep}).view({-1, 1})}, 1);
    int nms_post = -1;
    if (opts.find("nms_post")!=opts.end() && (nms_post = opts["nms_post"])<proposals.size(0)){
      proposals = proposals.index({Slice(None, nms_post)});
    }
    return proposals;
  }

  torch::Tensor RPNHeadImpl::forward_test(std::vector<torch::Tensor> feats, ImgData &img_data){
    auto rpn_outs = forward(feats);
    auto cls_out_list=std::get<0>(rpn_outs), bbox_out_list=std::get<1>(rpn_outs);
    auto anchor_list = _anchor_generator->get_anchors(get_grid_size(feats));
    cls_out_list = batch_permute(cls_out_list, {0, 2, 3, 1});
    bbox_out_list = batch_permute(bbox_out_list, {0, 2, 3, 1});
    cls_out_list = batch_reshape(cls_out_list, {-1, _class_channels});
    bbox_out_list = batch_reshape(bbox_out_list, {-1, 4});
    anchor_list = batch_reshape(anchor_list, {-1, 4}); 
    /*************** get proposals *************/
    return get_proposals(anchor_list, cls_out_list, bbox_out_list,
			 img_data.img_shape, _test_opts);
  }

}
