#ifndef ATSS_HEAD_HPP
#define ATSS_HEAD_HPP

#include <json.hpp>


namespace atss
{
  using json = nlohmann::json;

  class ATSSHeadImpl : public torch::nn::Module
  {
  public:
    ATSSHeadImpl();
    ATSSHeadImpl(const json &opts);
  private:

  };
  
  
}


#endif
