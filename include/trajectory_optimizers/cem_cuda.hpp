#ifndef CEM_CUDA_HPP
#define CEM_CUDA_HPP
#include "systems/enhanced_system.hpp"

#include <vector>
// #include <random>
#include <utility>
#include "iostream"
// #include <algorithm>
// #include <chrono>   
#include <stdio.h>

#include "cuda.h"
// #include "cuda_dep.h"
#include <curand.h>
#include <curand_kernel.h>
#include <algorithm>

//#include <thrust/sort.h>
//#include <thrust/device_ptr.h>
//#include <thrust/sequence.h>
//#include <thrust/device_vector.h>
//#include <device_launch_parameters.h>   // this is required in Windows. But it should be deleted in Linux

#include "trajectory_optimizers/cem.hpp"
#define DIM_STATE 4
#define DIM_CONTROL 1
#define NOBS 7

namespace trajectory_optimizers{
    class CEM_CUDA : public CEM{
        public:

            CEM_CUDA(enhanced_system_t* model, unsigned int number_of_samples, unsigned int number_of_t,
                unsigned int number_of_elite, double converge_r, 
                double control_means, double control_stds, 
                double time_means, double time_stds, double max_duration,
                double integration_step, double* loss_weights, unsigned int max_iteration, bool verbose, double step_size)
                : CEM(model, number_of_samples, number_of_t,
                    number_of_elite, converge_r, 
                    control_means, control_stds, 
                    time_means, time_stds, max_duration,
                    integration_step, loss_weights, max_iteration, verbose, step_size)
                {}

            CEM_CUDA(enhanced_system_t* model, unsigned int num_of_problems, unsigned int number_of_samples, unsigned int number_of_t,
                unsigned int number_of_elite,  double converge_r,
                std::vector<std::vector<double>>& _obs_list,
                double control_means, double control_stds, 
                double time_means, double time_stds, double max_duration,
                double integration_step, double* loss_weights, unsigned int max_iteration, bool verbose, double step_size);
                /*: CEM(model, number_of_samples, number_of_t,
                    number_of_elite, converge_r, 
                    control_means, control_stds, 
                    time_means, time_stds, max_duration,
                    integration_step, loss_weights, max_iteration, verbose, step_size) */
                
            ~CEM_CUDA(){
                delete weight;
                
                // delete best_ut;
                cudaFree(d_best_ut);
                cudaFree(d_temp_state);
                cudaFree(d_deriv);
                cudaFree(d_control);

                cudaFree(d_time);
                cudaFree(d_mean_time);
                cudaFree(d_mean_control);
                cudaFree(d_std_control);
                cudaFree(d_std_time);

                cudaFree(d_loss);
                cudaFree(d_top_k_loss);

                cudaFree(d_loss_ind);
                delete loss_ind;
                delete loss;
                delete obs_list;

                cudaFree(d_obs_list);
                cudaFree(d_active_mask);

                cudaFree(d_start_state);
                cudaFree(d_goal_state);
                cudaFree(devState);

            };

            unsigned int get_control_dimension();
            
            unsigned int get_num_step();

            // (state, goal) -> u, update mu_u and std_u
            virtual void solve(const double* start, const double* goal, double *best_u, double *best_t);
            double* weight;
            int NP;
            int NS;
            int N_ELITE;
            int NT;
            int it_max;
            double mu_u0;
            double std_u0;
            double mu_t0;
            double std_t0;
            double max_duration;
            int s_dim;
            int c_dim;
            double dt;

            bool verbose;
                // states for rolling
                
                // util variables for update statistics
            double converge_radius;
            // Cartpole();
            curandState* devState;


        protected:
            //enhanced_system_t *system;
            enhanced_system_t *system;
            double *d_temp_state, *d_control, *d_deriv, *d_time;
            double *d_mean_time, *d_mean_control, *d_std_control, *d_std_time;
            double *d_loss, *loss, *d_top_k_loss;
            int *d_loss_ind, *loss_ind;
            double /* *best_ut,*/ *d_best_ut;
            // for obstacles
            double* d_obs_list, *obs_list;
            bool* d_active_mask;
            std::vector<std::pair<double, int>> loss_pair;

            // for multi-start-goal
            double *d_start_state, *d_goal_state;
    };
}

#endif
