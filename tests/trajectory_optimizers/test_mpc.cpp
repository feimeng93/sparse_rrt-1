#include <iostream>
#include <vector>
#include "systems/cart_pole_obs.hpp"
#include "systems/two_link_acrobot_obs.hpp"
#include "systems/quadrotor_obs.hpp"
#include "systems/car_obs.hpp"
#include "trajectory_optimizers/cem.hpp"
// #include "trajectory_optimizers/cem_config.hpp"

using namespace std;
void test_acrobot(){
    std::vector<std::vector<double>> obs_list;
    double width = 5;
    obs_list.push_back(std::vector<double> {100.0, 200.0});
    enhanced_system_t* model = new two_link_acrobot_obs_t(obs_list, width);
    double loss_weights[4] = {1, 1, 0.3, 0.3};
    int ns = 3,
        nt = 2,
        ne = 2,
        max_it = 2;
    double converge_r = 1e-2,
        mu_t = 0.1,
        std_t = 0.2,
        t_max = 0.5,
        dt = 2e-2,
        step_size = 0;
    double std_u[1] = {4};
    double mu_u[1] = {0.1};
    trajectory_optimizers::CEM cem(model, ns, nt,               
                    ne, converge_r, 
                    &mu_u[0], &std_u[0], 
                    mu_t, std_t, t_max, 
                    dt, loss_weights, max_it, true, step_size);
    // double start[4] = {0, 0, 0, 0};
    // double goal[4] = {-0.30316862,  0.66820239, -1.04260097,  2.47421365};

    double start[4] = {-0.30316862,  0.66820239, -1.04260097,  2.47421365};
    double goal[4] = {-0.42044061,  0.96072684, -0.84960626,  2.32958837};

    // double start[4] = {-0.42044061,  0.96072684, -0.84960626,  2.32958837};
    // double goal[4] = {-0.48999742,  1.20535017, -0.02984635,  0.98378645};
  
    // double start[4] = {-2.60008393, -0.8919529 , -1.00603224,  2.0641629 };
    // double goal[4] = {-2.81999517, -0.29260973, -0.47518419,  1.90647392};
    
    double* u = new double[nt];
    double* t = new double[nt];
    cem.solve(start, goal, u, t);

    for(int ti = 0; ti < nt; ti++){
        cout << "control:\t" << u[ti] <<"\ttime:\t" << t[ti] << endl;
    }

    double state[4] = {start[0], start[1], start[2], start[3]};
    
    double current_loss = 1e3;
    for(int ti = 0; ti < nt; ti++){
        model -> propagate(state, 
                            4, 
                            &u[ti],
                            1,
                            (int)(t[ti] / dt),
                            state,
                            dt);
        current_loss = model->get_loss(state, goal, loss_weights);
        cout << (int)(t[ti] / dt) << endl;
        if (current_loss < converge_r){
            cout <<"reached"<< endl;
            break;
        } else {
            cout << "current_loss:" << current_loss<<endl;
        }
    }
    cout <<"final:"<< endl;
    for(int i = 0; i < 4; i++){
        cout << state[i] << ",";
    }

    cout << endl << current_loss << "\nreference:"<< endl;
    for(int i = 0; i < 4; i++){
        cout << goal[i] << ",";
    }

    cout << endl;
    // cem.solve(start, goal, u, t);
    // cout << endl << "second time"<< endl;

}

void test_cartpole(){
    std::vector<std::vector<double>> obs_list;
    double width = 5;
    obs_list.push_back(std::vector<double> {100.0, 200.0});
    enhanced_system_t* model = new cart_pole_obs_t(obs_list, width);
  
    double loss_weights[4] = {1, 0.5, 1, 0.5};
    int ns = 128,
        nt = 5,
        ne = 4,
        max_it = 10;
    double converge_r = 1e-2,
        mu_t = 5e-2,
        std_t = 1e-1,
        t_max = 0.1,
        dt = 2e-3,
        step_size = 0.75;
    double *mu_u = new double[1]{0.0},
           *std_u = new double[1]{100.};
    trajectory_optimizers::CEM cem(model, ns, nt,               
                    ne, converge_r, 
                    mu_u, std_u, 
                    mu_t, std_t, t_max, 
                    dt, loss_weights, max_it, true, step_size);

    double start[4] = {-10.2252733 ,   0.        ,   2.77379051,   0.        };
    double goal[4] =  {-10.78999025,  -4.11524938,   2.57380031,  -1.46940524};
    // double start[4] = {-10.78999025,  -4.11524938,   2.57380031,  -1.46940524};
    // double goal[4] = { -11.40025604,  -0.75909739,   2.24511758,  -1.2173847 };

    // double start[4] =  {-11.40025604,  -0.75909739,   2.24511758,  -1.2173847};
    // double goal[4] = {-11.39767163,   0.8476704 ,   2.14155191,  -1.14537533}; 
    
    double* u = new double[nt];
    double* t = new double[nt];
    cem.solve(start, goal, u, t);

    for(int ti = 0; ti < nt; ti++){
        cout << "control:\t" << u[ti] <<"\ttime:\t" << t[ti] << endl;
    }

    double state[4] = {start[0], start[1], start[2], start[3]};
    
    double current_loss = 1e3;
    for(int ti = 0; ti < nt; ti++){
        model -> propagate(state, 
                            4, 
                            &u[ti],
                            1,
                            (int)(t[ti] / dt),
                            state,
                            dt);
        current_loss = model->get_loss(state, goal, loss_weights);
        cout << (int)(t[ti] / dt) << endl;
        if (current_loss < converge_r){
            cout <<"reached"<< endl;
            break;
        } else {
            cout << "current_loss:" << current_loss<<endl;
        }
    }
    cout <<"final:"<< endl;
    for(int i = 0; i < 4; i++){
        cout << state[i] << ",";
    }

    cout << endl << current_loss << "\nreference:"<< endl;
    for(int i = 0; i < 4; i++){
        cout << goal[i] << ",";
    }

    cout << endl;
    // cem.solve(start, goal, u, t);
    // cout << endl << "second time"<< endl;

}

void test_quadrotor(){
    // std::vector<std::vector<double>> obs_list;
    // double width = 1;
    // obs_list.push_back(std::vector<double> {1.0, 2.0, 3.0});
    // enhanced_system_t* model = new quadrotor_obs_t(obs_list, width);
    enhanced_system_t* model = new quadrotor_obs_t();
    double loss_weights[13] = {1, 1, 1, 
                               0.3, 0.3, 0.3, 0.3,
                               0.3, 0.3, 0.3,
                               0.3, 0.3, 0.3};
    int ns = 32,
        nt = 3,
        ne = 4,
        max_it = 20;
    double converge_r = 0.1,
           *mu_u = new double[4]{-10, 0, 0, 0},
           *std_u = new double[4]{15, 1, 1, 1},
           mu_t = 0.1,
           std_t = 0.2,
           t_max = 0.5,
           dt = 2e-3,
           step_size = 1;
    trajectory_optimizers::CEM cem(model, ns, nt,               
                    ne, converge_r, 
                    mu_u, std_u, 
                    mu_t, std_t, t_max, 
                    dt, loss_weights, max_it, true, step_size);
    
    // double start[4] = {0, 0, 0, 0};
    // double goal[4] = {-0.30316862,  0.66820239, -1.04260097,  2.47421365};

    double start[13] = {-2.87136308, -2.54909565,  2.8230838 ,  0.        ,  0.        ,
                        0.        ,  1.        ,  0.        ,  0.        ,  0.        ,
                        0.        ,  0.        ,  0.        };
    double goal[13] = {-2.78339655, -2.50900839,  2.35692668, -0.0697973 ,  0.18299579,
                        0.13823697,  0.97084124,  0.46727852,  0.22953034, -1.        ,
                        -0.20080119,  0.52646407,  0.39769656};
    
    double* u = new double[4*nt];
    double* t = new double[nt];
    cem.solve(start, goal, u, t);

    for(int ti = 0; ti < nt; ti++){
        cout << "control:\t" << u[ti*4]<<","  << u[ti*4+1]<<","  << u[ti*4+2]<<","  << u[ti*4+3]<<"," <<"\ttime:\t" << t[ti] << endl;
    }


    double* state = new double[13]();
    for(int i = 0;i < 13; i++){
        state[i] = start[i];
    }
    
    double current_loss = 1e3;
    for(int ti = 0; ti < nt; ti++){
        model -> propagate(state, 
                            13, 
                            &u[ti*4],
                            4,
                            (int)(t[ti] / dt),
                            state,
                            dt);
        current_loss = model->get_loss(state, goal, loss_weights);
        cout << (int)(t[ti] / dt) << endl;
        if (current_loss < converge_r){
            cout <<"reached"<< endl;
            break;
        } else {
            cout << "current_loss:" << current_loss<<endl;
        }
    }
    cout <<"final:"<< endl;
    for(int i = 0; i < 13; i++){
        cout << state[i] << ",";
    }
    cout << endl;

    cout << endl << current_loss << "\nreference:"<< endl;
    for(int i = 0; i < 13; i++){
        cout << goal[i] << ",";
    }

    cout << endl;
    // cem.solve(start, goal, u, t);
    // cout << endl << "second time"<< endl;
    delete state;
}

void test_car(){
    std::vector<std::vector<double>> obs_list;
    double width = 8;
    obs_list.push_back(std::vector<double> {100.0, 200.0});
    // enhanced_system_t* model = new quadrotor_obs_t(obs_list, width);
    enhanced_system_t* model = new car_obs_t(obs_list, width);
    double loss_weights[13] = {1, 1};
    int ns = 1024,
        nt = 4,
        ne = 32,
        max_it = 5;
    double converge_r = 0.1,
           *mu_u = new double[2]{1, 0},
           *std_u = new double[2]{2, 0.5},
           mu_t = 0.25,
           std_t = 0.25,
           t_max = 0.5,
           dt = 2e-3,
           step_size = 1;
    trajectory_optimizers::CEM cem(model, ns, nt,               
                    ne, converge_r, 
                    mu_u, std_u, 
                    mu_t, std_t, t_max, 
                    dt, loss_weights, max_it, true, step_size);
    
    double start[model->get_state_dimension()] = {13.85253973, -16.09070246,   2.29143637};
    double goal[model->get_state_dimension()] = {12.92493622, -16.7003847, 2.79020587};

    // [[ 13.85253973 -16.09070246   2.29143637]
//  [ 12.92493622 -16.7003847    2.79020587]
    
    double* u = new double[nt*nt];
    double* t = new double[nt*nt];
    cem.solve(start, goal, u, t);

    for(int ti = 0; ti < nt; ti++){
        cout << "control:\t" << u[ti]<<","  << u[ti+1]<<","  << u[ti+2]<<","  << u[ti+3]<<"," <<"\ttime:\t" << t[ti] << endl;
    }


    double* state = new double[13]();
    for(int i = 0;i < 13; i++){
        state[i] = start[i];
    }
    
    double current_loss = 1e3;
    for(int ti = 0; ti < nt; ti++){
        model -> propagate(state, 
                            model->get_state_dimension(), 
                            &u[ti],
                            model->get_control_dimension(),
                            (int)(t[ti] / dt),
                            state,
                            dt);
        current_loss = model->get_loss(state, goal, loss_weights);
        cout << (int)(t[ti] / dt) << endl;
        if (current_loss < converge_r){
            cout <<"reached"<< endl;
            break;
        } else {
            cout << "current_loss:" << current_loss<<endl;
        }
    }
    cout <<"final:"<< endl;
    for(int i = 0; i < model->get_state_dimension(); i++){
        cout << state[i] << ",";
    }
    cout << endl;

    cout << endl << current_loss << "\nreference:"<< endl;
    for(int i = 0; i < model->get_state_dimension(); i++){
        cout << goal[i] << ",";
    }

    cout << endl;
    // cem.solve(start, goal, u, t);
    // cout << endl << "second time"<< endl;
    delete state;
}

int main(){
    // test_car();
    test_quadrotor();
    return 0;
}