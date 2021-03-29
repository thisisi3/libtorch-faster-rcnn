#ifndef TRAINER_HPP
#define TRAINER_HPP

#include <json.hpp>
#include <data.hpp>
#include <detectors.hpp>

namespace frcnn
{
  /*
    It constructs a model and then trains the model. It prints out basic training information like:
    timestamp, epoch#, iter/tot_iter, lr, eta, losses.
  **/
  class BasicTrainer
  {
  public:
    BasicTrainer(const json &opts);
    void train();
  private:
    void warmup_lr();
    void set_lr(float lr);
    float get_lr();
    torch::Tensor sum_loss(std::map<std::string, torch::Tensor> &loss_map);

    // private members
    FasterRCNN _model{nullptr};
    std::shared_ptr<CocoDataset> _dataset{nullptr};
    std::shared_ptr<torch::optim::SGD> _optimizer{nullptr};
    std::set<int> _decay_epochs;
    float _warmup_start;
    float _warmup_steps;
    int _total_epochs;
    int _save_ckpt_period;
    int _log_period;
    json _opts;

    torch::Device _device;
    std::string _work_dir;
    float _epoch_lr{-1};
    ProgressTracker _pg_tracker;  // track train process
  };

}

#endif
