#include <trainer.hpp>
#include <utils.hpp>
#include <iostream>
#include <fstream>
#include <json.hpp>

using json = nlohmann::json;
int main(int argc, char **argv){
  auto parser = frcnn::ArgumentParser("Train detection model");
  parser.add_argument("config", "config in json format")
    .add_option("gpu", false, "Provide GPU id, will choose CPU if not provided")
    .add_option("work-dir", true, "work directory");
  try{
    parser.parse(argc, argv);
    std::string cfg_json = parser.parsed_args[0];
    auto bslash_idx = cfg_json.rfind('/');
    std::string cfg_name = bslash_idx == std::string::npos ? cfg_json : cfg_json.substr(bslash_idx+1);
    json cfg;
    std::ifstream(cfg_json) >> cfg;
    cfg["work_dir"] = parser.parsed_opts["work-dir"][0];
    // copy config file to work_dir
    std::ofstream(cfg["work_dir"].get<std::string>() + "/" + cfg_name) << std::ifstream(cfg_json).rdbuf();
    if (parser.parsed_opts.find("gpu")!=parser.parsed_opts.end()){
      cfg["gpu"] = std::stol(parser.parsed_opts["gpu"].front());
    }
    auto trainer = frcnn::BasicTrainer(cfg);
    trainer.train();
    
  } catch (std::exception &e){
    std::cout << std::string("can not train model due to: ") + e.what() << std::endl;
  }
}
