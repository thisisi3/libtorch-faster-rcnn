#include <json.hpp>
#include <utils.hpp>
#include <torch/torch.h>
#include <detectors.hpp>
#include <data.hpp>

using json = nlohmann::json;

int main(int argc, char **argv){
  auto parser = frcnn::ArgumentParser("Inference");
  parser.add_argument("config and ckpt", "config in json format, ckpt in pt format")
    .add_option("gpu", false, "Provide GPU id, will choose CPU if not provided")
    .add_option("out", true, "output file");

  try{
    parser.parse(argc, argv);
    frcnn::ASSERT(parser.parsed_args.size()==2, "must provide config and ckpt as argument");
    std::string cfg_json = parser.parsed_args[0];
    std::string ckpt = parser.parsed_args[1];
    json cfg;
    std::ifstream cfg_if(cfg_json);
    cfg_if >> cfg;
    auto device = torch::Device(torch::kCPU);
    if (parser.parsed_opts.find("gpu")!=parser.parsed_opts.end()){
      device = torch::Device(torch::kCUDA, std::stoi(parser.parsed_opts["gpu"][0]));
    }
    auto model_opts = cfg["model"];
    auto model = frcnn::FasterRCNN(model_opts["backbone"],
				   model_opts["neck"],
				   model_opts["rpn_head"],
				   model_opts["rcnn_head"]);
    torch::load(model, ckpt);
    model->eval();
    model->to(device);
    auto dataset = std::make_shared<frcnn::CocoDataset>(cfg["data"]["test_imgs"].get<std::string>(),
							cfg["data"]["test_ann"].get<std::string>(),
							cfg["data"]["test_transforms"]);
    auto coco_ann = dataset->coco_ann();
    frcnn::ProgressTracker pg_tracker(1, dataset->size().value());
    auto loader_opts = torch::data::DataLoaderOptions().batch_size(1)
      .workers(cfg["data"]["test_workers"].get<int>());
    auto dataloader = torch::data::make_data_loader<torch::data::samplers::SequentialSampler>
      (std::move(*dataset), loader_opts);
    json res_json = json::array();
    torch::NoGradGuard no_grad;

    for(auto &img_datas : *dataloader){
      auto img_data = img_datas[0];
      img_data.to(device);
      auto det_res = model->forward_test(img_data.img_tsr, img_data);
      auto det_bboxes=std::get<0>(det_res), det_scores=std::get<1>(det_res), det_labels=std::get<2>(det_res);
      det_bboxes = frcnn::xyxy2xywhcoco(det_bboxes) / img_data.scale_factor;
      for(int i=0; i<det_bboxes.size(0); i++){
	auto bbox=det_bboxes[i];
	auto score=det_scores[i].item<float>();
	auto label=det_labels[i].item<long>();
	json cur_res_json = {
	  {"image_id", img_data.img_id},
	  {"bbox", {bbox[0].item<float>(), bbox[1].item<float>(), bbox[2].item<float>(), bbox[3].item<float>()}},
	  {"score", score},
	  {"category_id", coco_ann.cidx2cid[label]}
	};
	res_json.push_back(cur_res_json);
      }
      pg_tracker.next_iter();
      if(pg_tracker.cur_iter() % 100 == 0){
	pg_tracker.progress_bar();
      }
      if(pg_tracker.cur_iter() == pg_tracker.total_iters()){
	pg_tracker.progress_bar();
      }
    }
    std::ofstream out(parser.parsed_opts["out"][0]);
    out << res_json;
    std::cout << pg_tracker.elapsed() <<", " << frcnn::ProgressTracker::secs2str(pg_tracker.elapsed()) << std::endl;
  }catch(std::exception &e){
    std::cout << std::string("can not inference model due to: ") + e.what() << std::endl;
  }
}
