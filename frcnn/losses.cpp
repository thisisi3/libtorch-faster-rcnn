#include <losses.hpp>



namespace frcnn{

  std::shared_ptr<Loss> build_loss(const json &opts){
    std::string type = opts["type"].get<std::string>();
    if (type=="L1Loss"){
      return std::make_shared<L1Loss>(opts);
    } else if (type=="GIoULoss"){
      return std::make_shared<GIoULoss>(opts);
    } else if (type=="CrossEntropyLoss"){
      return std::make_shared<CrossEntropyLoss>(opts);
    } else if (type=="BinaryCrossEntropyLoss"){
      return std::make_shared<BinaryCrossEntropyLoss>(opts);
    } else {
      throw std::runtime_error("not supported loss type: "+type);
    }
    return nullptr;
  }


  //
  // L1 loss
  //
  L1Loss::L1Loss(double loss_weight)
    : _loss_weight(loss_weight)
  {}
  L1Loss::L1Loss(const json &opts)
    : _loss_weight(opts["loss_weight"].get<double>())
  { }
  torch::Tensor L1Loss::forward(torch::Tensor pred, torch::Tensor target, double avg_factor){
    return (pred - target).abs().sum() / avg_factor * _loss_weight;
  }

  //
  // GIoU loss
  //
  GIoULoss::GIoULoss(double loss_weight)
    : _loss_weight(loss_weight)
  { }
  GIoULoss::GIoULoss(const json &opts)
    : _loss_weight(opts["loss_weight"].get<double>())
  { }
  // assume both pred and target represent a same amount of bboxes                                                       
  torch::Tensor GIoULoss::forward(torch::Tensor pred, torch::Tensor target, double avg_factor){
    return (1 - giou(pred, target)).sum() / avg_factor * _loss_weight;
  }

  //
  // CrossEntropy loss
  //
  CrossEntropyLoss::CrossEntropyLoss(double loss_weight)
    : _loss_weight(loss_weight)
  { }
  CrossEntropyLoss::CrossEntropyLoss(const json &opts)
    : _loss_weight(opts["loss_weight"].get<double>())
  { }
  torch::Tensor CrossEntropyLoss::forward(torch::Tensor pred, torch::Tensor target, double avg_factor) {
    return torch::nn::functional::cross_entropy
      (pred, target, torch::nn::CrossEntropyLossOptions().reduction(torch::kSum)) / avg_factor * _loss_weight;
  }

  //
  // BCE loss
  //
  BinaryCrossEntropyLoss::BinaryCrossEntropyLoss(double loss_weight)
    : _loss_weight(loss_weight)
  { }
  BinaryCrossEntropyLoss::BinaryCrossEntropyLoss(const json &opts)
    : _loss_weight(opts["loss_weight"].get<double>())
  { }

  torch::Tensor BinaryCrossEntropyLoss::forward(torch::Tensor pred, torch::Tensor target, double avg_factor) {
    auto opts = torch::nn::functional::BinaryCrossEntropyWithLogitsFuncOptions().reduction(torch::kSum);
    return torch::nn::functional::binary_cross_entropy_with_logits
      (pred, target, opts) / avg_factor * _loss_weight;
  }





}
