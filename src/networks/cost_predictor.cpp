#ifndef COST_PREDICTOR
#include "networks/cost_predictor.hpp"
#endif
#include <string>

namespace networks{
    cost_predictor_t::cost_predictor_t(std::string network_weights_path) : network_t(network_weights_path){
        if(network_weights_path.length() != 0){
            network_torch_module_ptr.reset(new torch::jit::script::Module(
                torch::jit::load(network_weights_path)));
        } else {
            network_torch_module_ptr.reset(new torch::jit::script::Module(
                torch::jit::load("/media/arclabdl1/HD1/Linjun/mpc-mpnet-py/mpnet/exported/output/costnet5000.pt")));
        }
    }

    cost_predictor_t::~cost_predictor_t(){
        network_torch_module_ptr.reset();
    }

    at::Tensor cost_predictor_t::forward(std::vector<torch::jit::IValue> mpnet_input_container){
        return network_torch_module_ptr -> forward(mpnet_input_container).toTensor();
    }

    double cost_predictor_t::predict_cost(enhanced_system_t* system, torch::Tensor env_vox_tensor, const double* state, double* goal_state){
        double* normalized_state = new double[system->get_state_dimension()];
        double* normalized_goal = new double[system->get_state_dimension()];
        system -> normalize(state, normalized_state);
        system -> normalize(goal_state, normalized_goal);
        torch::Tensor state_goal_tensor = torch::ones({1, 8}).to(at::kCUDA); 
        std::vector<torch::jit::IValue> input_container;

        // set value state_goal with dim 1 x 8
        for(unsigned int si = 0; si < system->get_state_dimension(); si++){
            state_goal_tensor[0][si] = normalized_state[si]; 
        }
        for(unsigned int si = 0; si < system->get_state_dimension(); si++){
            state_goal_tensor[0][si + system->get_state_dimension()] = normalized_goal[si]; 
        }
        input_container.push_back(state_goal_tensor);
        input_container.push_back((env_vox_tensor));
        at::Tensor output = this -> forward(input_container);
        return output[0][0].item<float>();
    }
}