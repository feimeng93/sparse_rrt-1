#ifndef MPNET_COST_HPP
#include "networks/mpnet_cost.hpp"
#endif
#include <string>

#define DEBUG
namespace networks{
    mpnet_cost_t::mpnet_cost_t(std::string network_weights_path, 
        std::string cost_predictor_weights_path, int num_sample) : mpnet_t(network_weights_path),
        num_sample(num_sample)
        {
            if(network_weights_path.length() == 0){
                network_weights_path = "/media/arclabdl1/HD1/Linjun/mpc-mpnet-py/mpnet/exported/output/mpnet5000.pt";
            } 
            network_torch_module_ptr.reset(new torch::jit::script::Module(
                    torch::jit::load(network_weights_path)));
            if(cost_predictor_weights_path.length() == 0){
                cost_predictor_weights_path = "/media/arclabdl1/HD1/Linjun/mpc-mpnet-py/mpnet/exported/output/costnet5000.pt";
            }
            cost_predictor_torch_module_ptr.reset(new torch::jit::script::Module(
                    torch::jit::load(cost_predictor_weights_path)));
    }

    mpnet_cost_t::~mpnet_cost_t(){
        network_torch_module_ptr.reset();
        cost_predictor_torch_module_ptr.reset();
    }

    at::Tensor mpnet_cost_t::forward(std::vector<torch::jit::IValue> input_container){
        return network_torch_module_ptr -> forward(input_container).toTensor();
    }

    at::Tensor mpnet_cost_t::forward_cost(std::vector<torch::jit::IValue> input_container){
        return cost_predictor_torch_module_ptr -> forward(input_container).toTensor();
    }

    void mpnet_cost_t::mpnet_sample(enhanced_system_t* system, torch::Tensor env_vox_tensor,
        const double* state, double* goal_state, double* neural_sample_state){
        double* normalized_state = new double[system->get_state_dimension()];
        double* normalized_goal = new double[system->get_state_dimension()];
        double* normalized_neural_sample_state = new double[system->get_state_dimension()];
        system -> normalize(state, normalized_state);
        system -> normalize(goal_state, normalized_goal);
        std::vector<torch::jit::IValue> input_container;
        torch::Tensor state_goal_tensor = torch::ones({1, 8}).to(at::kCUDA); 

        // set value state_goal with dim 1 x 8
        for(unsigned int si = 0; si < system->get_state_dimension(); si++){
            state_goal_tensor[0][si] = normalized_state[si];    
        }
        for(unsigned int si = 0; si < system->get_state_dimension(); si++){
            state_goal_tensor[0][si + system->get_state_dimension()] = normalized_goal[si]; 
        }

        // for multiple sampling
        at::Tensor state_goal_tensor_expand = state_goal_tensor.repeat({num_sample, 1});
        at::Tensor env_vox_tensor_expand = env_vox_tensor.repeat({num_sample, 1, 1, 1});
        // 
        #ifdef DEBUG
        // for(unsigned int si = 0; si < 32; si++){
        //     std::cout << env_vox_tensor[0][0][0][si]<< std::endl; 
        // }
        #endif
        input_container.push_back(state_goal_tensor_expand);
        input_container.push_back(env_vox_tensor_expand);
        at::Tensor output = this -> forward(input_container);
        for(unsigned int di = 0; di < num_sample; di++){
            for(unsigned int si = 0; si < system->get_state_dimension(); si++){
                state_goal_tensor_expand[di][si] = output[di][si]; 
            }
        }
        at::Tensor predicted_costs = this -> forward_cost(input_container);
        at::Tensor best_index_tensor = at::argmin(predicted_costs);
        unsigned int best_index = best_index_tensor.item<int>();
        // int best_index = 0;
        for(unsigned int si = 0; si < system->get_state_dimension(); si++){
            normalized_neural_sample_state[si] = output[best_index][si].item<double>();
        }
       
        system -> denormalize(normalized_neural_sample_state, neural_sample_state);
        delete normalized_state;
        delete normalized_goal;
        delete normalized_neural_sample_state;
    }

}