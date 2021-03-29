#include <detectors.hpp>

namespace frcnn
{
  FasterRCNNImpl::FasterRCNNImpl(const json &backbone_opts,
				 const json &fpn_opts,
				 const json &rpn_opts,
				 const json &rcnn_opts)
    : _backbone_opts(backbone_opts),
      _fpn_opts(fpn_opts),
      _rpn_opts(rpn_opts),
      _rcnn_opts(rcnn_opts)
  {
    // building backbones is different as it allows different types of resnet
    _backbone = build_backbone(backbone_opts); 
    _neck = FPN(fpn_opts);
    _rpn_head = RPNHead(rpn_opts);
    _rcnn_head = RCNNHead(rcnn_opts);

    // weight initialization is included in constructor for all modules except for
    // initializing backbone with ImageNet pretrained weight
    std::string pretrained = backbone_opts["pretrained"].get<std::string>();
    if (pretrained!="None"){
      std::cout << "loading weights for backbone...\n";
      torch::load(_backbone, pretrained);
    }

    register_module("backbone", _backbone);
    register_module("neck", _neck);
    register_module("rpn_head", _rpn_head);
    register_module("rcnn_head", _rcnn_head);
  }
  
  // return a map/dict of losses
  std::map<std::string, torch::Tensor> FasterRCNNImpl::forward_train
  (torch::Tensor img_tsr, ImgData &img_data){
    auto feats = _backbone->forward(img_tsr);
    feats = _neck->forward(feats);
    auto rpn_outs = _rpn_head->forward_train(feats, img_data);
    auto rpn_cls_loss=std::get<0>(rpn_outs), rpn_bbox_loss=std::get<1>(rpn_outs), proposals=std::get<2>(rpn_outs);
    auto rcnn_outs = _rcnn_head->forward_train(feats, proposals, img_data);
    auto rcnn_cls_loss = std::get<0>(rcnn_outs), rcnn_bbox_loss = std::get<1>(rcnn_outs);
    return {{"rpn_cls_loss", rpn_cls_loss},
	{"rpn_bbox_loss", rpn_bbox_loss}, {"rcnn_cls_loss", rcnn_cls_loss},{"rcnn_bbox_loss", rcnn_bbox_loss}};
  }
  
  // return det_bboxes, det_scores, det_labels                                                                           
  std::tuple<torch::Tensor, torch::Tensor, torch::Tensor> FasterRCNNImpl::forward_test
  (torch::Tensor img_tsr, ImgData &img_data){
    auto feats = _backbone->forward(img_tsr);
    feats = _neck->forward(feats);
    auto proposals = _rpn_head->forward_test(feats, img_data);
    auto det_res = _rcnn_head->forward_test(feats, proposals, img_data);
    return det_res;
  }
  
}
