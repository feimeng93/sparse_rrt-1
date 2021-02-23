#ifndef COST_PREDICTOR_HPP
#define COST_PREDICTOR_HPP

#ifndef NETWORK_HPP
#include "networks/network.hpp"
#endif

#ifndef SPARSE_ENHANCED_SYSTEM_HPP
#include "systems/enhanced_system.hpp"
#endif

namespace networks{
    class cost_predictor_t : public network_t
    {
    public:
        cost_predictor_t(std::string network_weights_path);
        at::Tensor forward(std::vector<torch::jit::IValue> mpnet_input_container);
        double predict_cost(enhanced_system_t* system, torch::Tensor env_vox_tensor, const double* state, double* goal_state);
        ~cost_predictor_t();
    protected:
        std::shared_ptr<torch::jit::script::Module> network_torch_module_ptr;
    };
}
#endif
