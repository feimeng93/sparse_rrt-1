#include "trajectory_optimizers/cem.hpp"

#include "systems/two_link_acrobot_obs.hpp"
#include "systems/quadrotor_obs.hpp"

#include "systems/distance_functions.h"

#include "motion_planners/deep_smp_mpc_sst.hpp"

#include "networks/mpnet.hpp"
#include "networks/mpnet_cost.hpp"

#include <torch/script.h>
#include <iostream>
#include <string>

using namespace std;

int main(){
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
    deep_smp_mpc_sst_t* planner;

    double in_start[4] = {0, 0, 0, 0};
    double in_goal[4] = {3.14, 0, 0}; 
    double in_radius = 2; 

    torch::Tensor obs = torch::zeros({1,1,32,32}).to(at::kCUDA);
    torch::NoGradGuard no_grad;

    planner = new deep_smp_mpc_sst_t(
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