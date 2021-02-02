#include "trajectory_optimizers/cem.hpp"

#include "systems/two_link_acrobot_obs.hpp"
#include "systems/quadrotor_obs.hpp"

#include "systems/distance_functions.h"

#include "motion_planners/mpc_mpnet.hpp"

#include "networks/mpnet.hpp"
#include "networks/mpnet_cost.hpp"

#include <torch/script.h>
#include <iostream>
#include <string>

using namespace std;

int test_acrobot(){
    std::vector<std::vector<double>> obs_list;
    double width = 5;
    obs_list.push_back(std::vector<double> {100.0, 200.0});
    
    
    enhanced_system_t* model = new two_link_acrobot_obs_t(obs_list, width);
    // enhanced_system_t* model = new quadrotor_obs_t();
    // initialize cem
    double loss_weights[4] = {1, 1, 0.3, 0.3};
    int ns = 1024,
        nt = 5,
        ne = 32,
        max_it = 20;
    double converge_r = 0.1,
        *mu_u = new double[1]{0},
        *std_u = new double[1]{4},
        mu_t = 0.1,
        std_t = 0.2,
        t_max = 0.5,
        dt = 2e-2,
        step_size = 0.5;
    trajectory_optimizers::CEM cem(model, ns, nt,               
                    ne, converge_r, 
                    mu_u, std_u, 
                    mu_t, std_t, t_max, 
                    dt, loss_weights, max_it, true, step_size);
    
    // initialzie mpnet
    networks::mpnet_cost_t mpnet(
        std::string(""),
        std::string(""),
        std::string(""),
        5, "cuda:0", 0.2, true);
    //  networks::mpnet_t mpnet(
    //     std::string("/media/arclabdl1/HD1/Linjun/mpc-mpnet-py/mpnet/exported/output/mpnet5000.pt"));
    mpc_mpnet_t* planner;

    double in_start[4] = {0, 0, 0, 0};
    double in_goal[4] = {3.14, 0, 0}; 
    double in_radius = 2; 

    torch::Tensor obs = torch::zeros({1,1,32,32}).to(at::kCUDA);
    torch::NoGradGuard no_grad;

    planner = new mpc_mpnet_t(
        in_start, in_goal,
        in_radius,
        model->get_state_bounds(),
        model->get_control_bounds(),
        two_link_acrobot_obs_t::distance,
        0,
        5e-1, 1e-2,
        &cem,
        &mpnet, 1, 1);
    planner -> step(model, 10, 50, dt);


    // const double start[4] = {-0.42044061,  0.96072684, -0.84960626,  2.32958837};
    // const double goal[4] = {-0.48999742,  1.20535017, -0.02984635,  0.98378645};
    // double * terminal = new double[model->get_state_dimension()];
    // double duration = 0;
    // planner -> steer(model, start, goal, &duration, terminal, dt);
    // planner -> neural_step(model, dt, obs, true, 0);

    return 0;
}


int test_quadrotor(){
    double width = 1;
    std::vector<std::vector<double>> obs_list;
    obs_list.push_back(std::vector<double> {4., 4., 4.});
    enhanced_system_t* model = new quadrotor_obs_t(obs_list, width);

    // initialize cem
    double loss_weights[13] = {1, 1, 1, 
                               0.3, 0.3, 0.3, 0.3,
                               0.3, 0.3, 0.3,
                               0.3, 0.3, 0.3};
    int ns = 1024,
        nt = 5,
        ne = 32,
        max_it = 20;
    double converge_r = 0.1,
           *mu_u = new double[4]{0, 0, 0, 0},
           *std_u = new double[4]{15, 1, 1, 1},
           mu_t = 0.1,
           std_t = 0.2,
           t_max = 0.5,
           dt = 2e-3,
           step_size = 0.5;
    trajectory_optimizers::CEM cem(model, ns, nt,               
                    ne, converge_r, 
                    mu_u, std_u, 
                    mu_t, std_t, t_max, 
                    dt, loss_weights, max_it, true, step_size);
    
    // initialzie mpnet
    networks::mpnet_cost_t mpnet(
        std::string(""),
        std::string(""),
        std::string(""),
        5, "cuda:0", 0.2, true);
    mpc_mpnet_t* planner;

    const double in_start[13] = {0, 0, 0, 
                                 0, 0, 0, 1,
                                 0, 0, 0,
                                 0, 0, 0,
                          };
    double in_goal[13] = {0, 0, 2, 
                          0, 0, 0, 1,
                          0, 0, 0,
                          0, 0, 0,
                          };;
    double in_radius = 3; 

    torch::Tensor obs = torch::zeros({1,32,32,32}).to(at::kCUDA);
    torch::NoGradGuard no_grad;

    planner = new mpc_mpnet_t(
        in_start, in_goal,
        in_radius,
        model->get_state_bounds(),
        model->get_control_bounds(),
        quadrotor_obs_t::distance,
        0,
        0.3, 0.3,
        &cem,
        &mpnet, 1, 1);

    // // Test system propagation
    // double const control[4] = {-15, 0, 0, 0};
    // double state[13] = {0, 0, 0, 
    //                       0, 0, 0, 0,
    //                       0, 0, 0,
    //                       0, 0, 0,
    //                       };
    // model->propagate(in_start, model->get_state_dimension(), 
    //                  control, model->get_control_dimension(), 
    //                  10, state, dt);
    // // std::copy(state, state + model->get_state_dimension(), std::ostream_iterator<double>(std::cout, ", "));
    // for(int i = 0; i < model->get_state_dimension(); i++){
    //     std::cout<<state[i]<<", ";
    // }
    // std::cout<<std::endl;

    // Test step function
    for(int i = 0; i < 300000; i++){
        planner -> step(model, 1, 200, dt);
        if(i % 1000 == 1){
            std::vector<std::vector<double>> solution_path;
            std::vector<std::vector<double>> controls;
            std::vector<double> costs;
            planner->get_solution(solution_path, controls, costs);
            std::cout << i <<"-th iteration, ";
            if (controls.size() > 0) {
                std::copy(costs.begin(), costs.end(), std::ostream_iterator<double>(std::cout, " "));
                std::cout<<", cost:"<< accumulate(costs.begin(), costs.end(), 0.0)<<std::endl;
            } else {
                std::cout<<std::endl;
            }

        }        
    }
    // const double start[4] = {-0.42044061,  0.96072684, -0.84960626,  2.32958837};
    // const double goal[4] = {-0.48999742,  1.20535017, -0.02984635,  0.98378645};
    // double * terminal = new double[model->get_state_dimension()];
    // double duration = 0;
    // planner -> steer(model, start, goal, &duration, terminal, dt);
    // planner -> neural_step(model, dt, obs, true, 0);

    return 0;
}

int main(){
    test_acrobot();
    return 0;
}