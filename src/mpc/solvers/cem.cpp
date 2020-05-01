#include "mpc/solvers/cem.hpp"

namespace solvers{
    void CEM::solve(double* start, double* goal, double* best_u, double* best_t){
        // initialize 
        // initialize control_dist: [nt * c_dim]
        std::normal_distribution<double>* control_dist = 
            new std::normal_distribution<double>[number_of_t * c_dim];
        for(int i = 0; i < number_of_t * c_dim; i++){
            mu_u[i] = mu_u0;
            std_u[i] = std_u0;
        }
        // initialize time_dist: [nt]
        std::normal_distribution<double>* time_dist = 
            new std::normal_distribution<double>[number_of_t];
        for(int i = 0; i < number_of_t; i++){
            mu_t[i] = mu_t0;
            std_t[i] = std_t0;
        }
        // initialize loss: set to <0, index>
        loss = std::vector<std::pair<double, int>>();
        // initialize states: set to starts
        for(int si = 0; si < number_of_samples; si++){
            for(int j = 0; j < s_dim; j++){
                states[si * s_dim + j] = start[j];
            }
            loss.push_back(std::make_pair(0., si));
        }
        // set early stop parameters
        double min_loss = 1e3; // to +inf
        int early_stop_count = 0;

        // double* best_u = new double[c_dim];
        // begin loop
        for(int it = 0; it < 9999; it++){
            // reset active_mask in every iteration
            for(int si = 0; si < number_of_samples; si++){
                active_mask[si] = true;
            }
            // sampling control and time
            for(int ti = 0; ti < number_of_t; ti++){
                // reset statistics parameters
                for(int ci = 0; ci < c_dim; ci++){
                    control_dist[ti * c_dim + ci] = 
                        std::normal_distribution<double>
                        (mu_u[ti * c_dim + ci], std_u[ti * c_dim + ci]);
                }
                time_dist[ti] = 
                        std::normal_distribution<double>
                        (mu_t[ti], std_u[ti]);
                
            }

            for(int si = 0; si < number_of_samples; si++){
                for (int ti = 0; ti < number_of_t; ti++){
                    // generate control samples
                    for(int ci = 0; ci < c_dim; ci++){
                        controls[si* number_of_t * c_dim + ti * c_dim + ci] = 
                            control_dist[ti](generator);
                    }
                    // generate duration samples and wrap to [0, max_duration]
                    time[si* number_of_t * ti] = 
                            time_dist[ti](generator);
                    if (time[ti] > max_duration){
                        time[ti] = 0;
                    } else if(time[ti] < 0){
                        time[ti] = 0;
                    }
                }
            }
            // propagation loops
            // TODO: Parallization?
            for(int si = 0; si < number_of_samples; si++){ // reset loss
                loss.at(si).first = 0; // loss: vector<double loss_value, int index>
                loss.at(si).second = si;
            }
            for(int ti=0; ti < number_of_t; ti++){ // time loop
                for(int si = 0; si < number_of_samples; si++){
                    if (active_mask[si]){
                        if (system -> propagate(&states[si * s_dim], 
                            s_dim, 
                            &controls[si * number_of_t * c_dim + ti * c_dim],
                            c_dim,
                            (int)(time[si * number_of_t + ti] / dt),
                            &states[si * s_dim],
                            dt)){// collision free
                                loss.at(si).first += time[si * number_of_t + ti];
                                // loss.at(si).first += system -> get_loss(
                                //     &states[si*s_dim], goal, weight
                                //     );
                                double current_sample_loss = system -> get_loss(
                                    &states[si*s_dim], goal, weight
                                    );
                                if (current_sample_loss < converge_radius){
                                    active_mask[si] = false;
                                }
                        }
                        else{ // collision
                            loss.at(si).first += 1e2;
                        }
                    }
                }
            }            
            for(int si = 0; si < number_of_samples; si++){ // terminal_loss
                loss.at(si).first = system -> get_loss(
                &states[si * s_dim], goal, weight);
            }
            
            //update statistics
            sort(loss.begin(), loss.end());

            for(int ti = 0; ti < number_of_t; ti++){
                for(int ci = 0; ci < c_dim; ci++){
                    sum_of_controls[ti * c_dim + ci] = 0;
                    sum_of_square_controls[ti * c_dim + ci] = 0;
                }
                sum_of_time[ti] = 0;
                sum_of_square_time[ti] = 0;
            }
            for(int si = 0; si < number_of_elite; si++){
                int index = loss.at(si).second;
                for(int ti = 0; ti < number_of_t; ti++){
                    for(int ci = 0; ci < c_dim; ci++){
                        double control_i = 
                            controls[index * number_of_t * c_dim + ti * c_dim + ci];
                        sum_of_controls[ti * c_dim + ci] += control_i;
                        sum_of_square_controls[ti * c_dim + ci] += control_i * control_i;
                    }
                    double time_i = time[index * number_of_t + ti];
                    sum_of_time[ti] += time_i;
                    sum_of_square_time[ti] += time_i * time_i;
                }
            }
            for(int ti = 0; ti < number_of_t; ti++){
                for(int ci = 0; ci < c_dim; ci++){
                    mu_u[ti * c_dim + ci] = sum_of_controls[ti * c_dim + ci] / number_of_elite;
                    std_u[ti * c_dim + ci] = sqrt(
                        sum_of_square_controls[ti * c_dim + ci] / 
                        number_of_elite - mu_u[ti * c_dim + ci] * mu_u[ti * c_dim + ci]
                        );
                }
                mu_t[ti] = sum_of_time[ti] / number_of_elite;
                std_t[ti] = sqrt(
                    sum_of_square_time[ti] / number_of_elite - 
                        mu_t[ti] * mu_t[ti]
                    );
            }
            // early stop checking
            if(loss.at(0).first < min_loss){
                min_loss = loss.at(0).first;
                for(int ti = 0; ti < number_of_t; ti++){
                    for(int ci = 0; ci < c_dim; ci++){
                        best_u[ti*c_dim + ci] = controls[loss.at(0).second * number_of_t * c_dim + ti * c_dim + ci];
                    }
                    best_t[ti] = time[loss.at(0).second * number_of_t + ti];
                }
                

            }
            else{
                early_stop_count += 1;
                if(early_stop_count >= it_max){
                    break;
                }
            }

            if(verbose){
                std::cout <<it<<"\tloss:"<< loss.at(0).first<<"\tminLoss:"<< min_loss;
                /*for(int si = 0; si < s_dim; si++){
                    std::cout <<<<"\tstates:\t" << states[loss.at(0).second * s_dim + si];
                }*/
                std::cout<< std::endl;
            }
            if(min_loss < converge_radius){
                    break;
            }
        }// end loop

    }

     std::vector<std::vector<double>> CEM::rolling(double* start, double* goal){
        // initialize
        for(int i = 0; i < s_dim; i++){
             current_state[i] = start[i];
        }

        for(int i = 0; i < number_of_t * c_dim; i++){
            mu_u[i] = mu_u0;
            std_u[i] = std_u0;
        }
        int early_stop_count = 0;
        std::vector<std::vector<double>> path; 
        double current_loss = 1e2;
        double* u = new double[number_of_t * c_dim];
        double* t = new double[number_of_t];
        // rolling
        while(current_loss > converge_radius && early_stop_count < it_max){
            CEM::solve(start, goal, u, t);
            if(! system -> propagate(current_state, s_dim, &u[0], c_dim, 
                (int)(t[0]/dt), current_state, dt)){
                    break;
            }
            current_loss = system -> get_loss(current_state, goal, weight);
            std::cout<<"current_loss:"<< current_loss<< std::endl;
            std::vector<double> path_node;
            for(int i = 0; i < s_dim; i++){
                path_node.push_back(current_state[i]);
            }
            path.push_back(path_node);
        }
        return path;
     }
    
}