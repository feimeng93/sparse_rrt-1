#ifndef MPNET_COST_HPP
#include "networks/mpnet_cost.hpp"
#endif
#include <string>

// #define DEBUG
namespace networks{
    mpnet_cost_t::mpnet_cost_t(std::string network_weights_path, 
        std::string cost_predictor_weights_path, int num_sample, std::string device_id, float refine_lr) : mpnet_t(network_weights_path),
        num_sample(num_sample), device_id(device_id), refine_lr(refine_lr)
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
            network_torch_module_ptr->to(torch::Device(device_id));;
            cost_predictor_torch_module_ptr->to(torch::Device(device_id));
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
        const double* state, double* goal_state, double* neural_sample_state, bool refine, float refine_threshold,
        bool using_one_step_cost, bool cost_reselection){
        double* normalized_state = new double[system->get_state_dimension()];
        double* normalized_goal = new double[system->get_state_dimension()];
        double* normalized_neural_sample_state = new double[system->get_state_dimension()];
        system -> normalize(state, normalized_state);
        system -> normalize(goal_state, normalized_goal);
        std::vector<torch::jit::IValue> input_container;
        torch::Tensor state_tensor = torch::ones({1, 4}).to(torch::Device(device_id)); 
        torch::Tensor goal_tensor = torch::ones({1, 4}).to(torch::Device(device_id)); 
        // set value state_goal with dim 1 x 8
        for(unsigned int si = 0; si < system->get_state_dimension(); si++){
            state_tensor[0][si] = normalized_state[si];    
        }
        for(unsigned int si = 0; si < system->get_state_dimension(); si++){
            goal_tensor[0][si] = normalized_goal[si]; 
        }
        
        at::Tensor state_tensor_expand = state_tensor.repeat({num_sample, 1}).to(torch::Device(device_id));
        at::Tensor goal_tensor_expand = goal_tensor.repeat({num_sample, 1}).to(torch::Device(device_id));

        torch::Tensor state_goal_tensor = at::cat({state_tensor_expand, goal_tensor_expand}, 1).to(torch::Device(device_id));
        // for multiple sampling
        at::Tensor env_vox_tensor_expand = env_vox_tensor.repeat({num_sample, 1, 1, 1}).to(torch::Device(device_id));
        input_container.push_back(state_goal_tensor);
        input_container.push_back(env_vox_tensor_expand);
        at::Tensor predicted_state_tensor = this -> forward(input_container).to(torch::Device(device_id));
        at::Tensor cost_input;
        at::Tensor predicted_costs;
        at::Tensor best_index_tensor;
        torch::Tensor predicted_state_var;
        // at::Tensor current_cost_to_go;
        if(refine){
            predicted_state_var = torch::autograd::Variable(predicted_state_tensor.clone()).detach().set_requires_grad(true);
        }else{
            predicted_state_var = predicted_state_tensor;
        }
        if(using_one_step_cost){
            cost_input = at::cat({state_tensor_expand, predicted_state_var}, 1).to(torch::Device(device_id));
            input_container.at(0) = cost_input;
            predicted_costs = this -> forward_cost(input_container).to(torch::Device(device_id));
            best_index_tensor = at::argmin(predicted_costs).to(torch::Device(device_id));
            
        } else {// using cost to go
            // current_cost_to_go = this -> forward_cost(input_container).to(torch::Device(device_id));
            cost_input = at::cat({predicted_state_var, goal_tensor_expand}, 1).to(torch::Device(device_id));
            input_container.at(0) = cost_input;
            predicted_costs = this -> forward_cost(input_container).to(torch::Device(device_id));
            best_index_tensor = at::argmin(predicted_costs).to(torch::Device(device_id));
        }           
        // refining goes here
        if(refine){
            predicted_costs.sum().backward();
            torch::Tensor refine_grad = predicted_state_var.grad().to(torch::Device(device_id));
            torch::Tensor refine_norm = at::norm(refine_grad, 2, 1, true);
            // std::cout<< refine_norm.sizes()<<std::endl;
            // for(int i = 0; i < num_sample; i++){
            //     std::cout<<"norm:\t"<<refine_norm[i].item<double>() << std::endl;
            // }
            // refine_grad /= refine_norm;
            if(using_one_step_cost){
                // predicted_state_tensor = predicted_state_tensor - refine_lr * refine * ((predicted_costs) > refine_threshold);
                predicted_state_tensor = predicted_state_tensor - refine_lr * refine_grad * (refine_norm > refine_threshold);
            } else{// using cost to go
                // predicted_state_tensor = predicted_state_tensor - refine_lr * refine * ((current_cost_to_go - predicted_costs) > refine_threshold);
                predicted_state_tensor = predicted_state_tensor - refine_lr * refine_grad * (refine_norm > refine_threshold);
                // predicted_state_tensor = predicted_state_tensor -
                //     refine_lr * refine_grad * (refine_norm > refine_threshold);

            }
            
        }
       
        if(cost_reselection){
            if(using_one_step_cost){
                cost_input = at::cat({state_tensor_expand, predicted_state_tensor}, 1).to(torch::Device(device_id));
            }
            else{
                cost_input = at::cat({predicted_state_tensor, goal_tensor_expand}, 1).to(torch::Device(device_id));
            }
            input_container.at(0) = cost_input;
            predicted_costs = this -> forward_cost(input_container).to(torch::Device(device_id));
            best_index_tensor = at::argmin(predicted_costs).to(torch::Device(device_id));
        }
        unsigned int best_index = best_index_tensor.item<int>();
        
        for(unsigned int si = 0; si < system->get_state_dimension(); si++){
            normalized_neural_sample_state[si] = predicted_state_tensor[best_index][si].item<double>();
        }
        system -> denormalize(normalized_neural_sample_state, neural_sample_state);
        delete normalized_state;
        delete normalized_goal;
        delete normalized_neural_sample_state;
    }

    

}


// torch::Tensor rand2= torch::autograd::Variable(rand.clone()).detach().set_requires_grad(true);  //rand is basically mpnet output, we convert it to variable that requires grad
// torch::Tensor dnet_input= torch::cat({voxel, ohot, rand2},1).to(torch::Device(device_id)); // making it as input for the cost-to-go function (dnet)
// inputs_dnet.push_back(dnet_input);
// torch::Tensor dnet = dnet.forward(inputs_dnet).toTensor();
// //dout is basically the predicted cost
// dout.backward(); // next you run backward pass so that the repair as the gradients in it
// torch::Tensor repair=rand2.grad();
// // this is how we project using the gradiants
// dout = dout.to(at::kCPU);
// auto dout_vec= dout.accessor<float,2>();
// if (((dReal)dout_vec[0][0])>0.3)
// {	
//     rand=rand-0.4*(repair);
// }
// 	rand = rand.to(at::kCPU);
// to_vector(rand, _randomConfig);
// setlimits(_randomConfig);