#include <iostream>
#include <vector>
#include "systems/cart_pole_obs.hpp"
#include "systems/two_link_acrobot_obs.hpp"
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
        mu_u = 0,
        std_u = 4,
        mu_t = 0.1,
        std_t = 0.2,
        t_max = 0.5,
        dt = 2e-2,
        step_size = 0;
    trajectory_optimizers::CEM cem(model, ns, nt,               
                    ne, converge_r, 
                    mu_u, std_u, 
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
        mu_u = 0,
        std_u = 100,
        mu_t = 5e-2,
        std_t = 1e-1,
        t_max = 0.1,
        dt = 2e-3,
        step_size = 0.75;
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

int main(){
    test_cartpole();
    return 0;
}