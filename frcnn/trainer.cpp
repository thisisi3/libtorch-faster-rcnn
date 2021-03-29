#include <trainer.hpp>


namespace frcnn
{
  BasicTrainer::BasicTrainer(const json &opts)
    : _device(torch::kCPU),
      _pg_tracker(0, 0)
  {
    _opts = opts;
    _dataset = std::make_shared<CocoDataset>(opts["data"]["train_imgs"].get<std::string>(),
					     opts["data"]["train_ann"].get<std::string>(),
					     opts["data"]["train_transforms"]);
    auto model_opts = opts["model"];
    // construct FasterRCNN object detector
    _model = FasterRCNN(model_opts["backbone"],
			model_opts["neck"],
			model_opts["rpn_head"],
			model_opts["rcnn_head"]);
    auto optimizer_opts = opts["optimizer"];
    ASSERT(optimizer_opts["type"].get<std::string>() == "SGD", "only support SGD optimizer");
    // construct SGD options for later construction of SGD optimizer
    auto optim_opts = torch::optim::SGDOptions(optimizer_opts["lr"].get<float>())
      .momentum(optimizer_opts["momentum"].get<float>())
      .weight_decay(optimizer_opts["weight_decay"].get<float>());
    _epoch_lr = optimizer_opts["lr"].get<float>();
    // construct SGD optimizer
    _optimizer = std::make_shared<torch::optim::SGD>(_model->parameters(), optim_opts);
    auto decay_epochs = opts["lr_opts"]["decay_epochs"].get<std::vector<int>>();
    _decay_epochs = std::set<int>(decay_epochs.begin(), decay_epochs.end());
    _warmup_start = opts["lr_opts"]["warmup_start"].get<float>();
    _warmup_steps = opts["lr_opts"]["warmup_steps"].get<float>();
    _total_epochs = opts["total_epochs"].get<int>();
    _save_ckpt_period = opts["save_ckpt_period"].get<int>();
    _log_period = opts["log_period"].get<int>();
    _work_dir = opts["work_dir"].get<std::string>();
    if (opts.find("gpu")!=opts.end()){
      _device = torch::Device(torch::kCUDA, opts["gpu"].get<int>());
    }
  }

  void BasicTrainer::train(){
    _pg_tracker = ProgressTracker(_total_epochs, _dataset->size().value());
    _model->to(_device);
    _model->train();
    std::cout << _model << std::endl;
    auto loader_opts = torch::data::DataLoaderOptions().batch_size(1)
      .workers(_opts["data"]["train_workers"].get<int>());
    auto dataloader = torch::data::make_data_loader<torch::data::samplers::RandomSampler>
      (std::move(*_dataset), loader_opts);
    // start to train model epoch by epoch
    for (int64_t epoch = 1; epoch <= _total_epochs; epoch++){
      // check if lr needs to be decayed
      if(_decay_epochs.find(epoch) != _decay_epochs.end()){
	_epoch_lr *= 0.1;
	set_lr(_epoch_lr);
      }
      // iterate over all image data
      for(auto &img_datas : *dataloader){
	// check if lr needs to be warmed up at the begining
	auto img_data = img_datas[0];
	img_data.to(_device);
	warmup_lr();
	auto model_loss = _model->forward_train(img_data.img_tsr, img_data);
	auto tot_loss = sum_loss(model_loss);
	model_loss["loss"] = tot_loss;
	_pg_tracker.track_loss(model_loss);
	_optimizer->zero_grad();
	tot_loss.backward();
	_optimizer->step();
	_pg_tracker.next_iter();
	if (_pg_tracker.cur_iter() % _log_period == 0){
	  _pg_tracker.track_lr(get_lr());
	  _pg_tracker.report_progress(std::cout);
	}
      }
      _pg_tracker.next_epoch();
      if(epoch % _save_ckpt_period == 0){
	torch::save(_model, _work_dir + "/epoch_" + std::to_string(epoch) + ".pt");
      }
    }
  }
  
  void BasicTrainer::warmup_lr(){
    auto iters = _pg_tracker.cur_iter();
    if (iters <= _warmup_steps){
      float lr = _warmup_start * _epoch_lr + (1-_warmup_start) * iters / _warmup_steps * _epoch_lr;
      set_lr(lr);
    }
  }

  void BasicTrainer::set_lr(float lr){
    for (auto &group : _optimizer->param_groups()){
      static_cast<torch::optim::SGDOptions&>(group.options()).lr(lr);
    }
  }

  float BasicTrainer::get_lr(){
    return static_cast<torch::optim::SGDOptions&>(_optimizer->param_groups()[0].options()).lr();
  }

  torch::Tensor BasicTrainer::sum_loss(std::map<std::string, torch::Tensor> &loss_map){
    auto tot_loss = torch::tensor(0, torch::TensorOptions().dtype(torch::kFloat32).device(_device));
    for (auto &loss : loss_map){
      tot_loss += loss.second;
    }
    return tot_loss;
  }
  
}
