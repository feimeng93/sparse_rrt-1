#include <iostream>
#include <vector>
#include "systems/mpc_system.hpp"
#include "systems/two_link_acrobot_obs_t.hpp"
#include "mpc/solvers/cem.hpp"

using namespace std;

int main(){
    std::vector<std::vector<double>> obs_list;
    double width = 1;
    obs_list.push_back(std::vector<double> {100.0, 200.0});
    system_t* model = new two_link_acrobot_obs_t(obs_list, width);
    // cout<<"hello"<<endl;
    double loss_weights[4] = {1, 1, 0.3, 0.3};

    int ns = 1024,
        nt = 5,
        ne = 32,
        max_it = 20;
    double coverge_r = 0.1,
        mu_u = 0,
        std_u = 4,
        mu_t = 0.05,
        std_t = 0.2,
        t_max = 0.5,
        dt = 2e-2;
    solvers::CEM cem(model, ns, nt,               
                    ne, coverge_r, 
                    mu_u, std_u, 
                    mu_t, std_t, t_max, 
                    dt, loss_weights, max_it, true);
    // double start[4] = {0, 0, 0, 0};
    // double goal[4] = {-0.30316862,  0.66820239, -1.04260097,  2.47421365};

    // double start[4] = {-0.30316862,  0.66820239, -1.04260097,  2.47421365};
    // double goal[4] = {-0.42044061,  0.96072684, -0.84960626,  2.32958837};

    // double start[4] = {-0.42044061,  0.96072684, -0.84960626,  2.32958837};
    // double goal[4] = {-0.48999742,  1.20535017, -0.02984635,  0.98378645};
  
    double start[4] = {-2.60008393, -0.8919529 , -1.00603224,  2.0641629 };
    double goal[4] = {-2.81999517, -0.29260973, -0.47518419,  1.90647392};
    
    double* u = new double[nt];
    double* t = new double[nt];
    cem.solve(start, goal, u, t);
    for(int ti = 0; ti < nt; ti++){
        cout << "control:\t" << u[ti] <<"\ttime:\t" << t[ti] << endl;
    }

    double state[4] = {start[0], start[1], start[2], start[3]};
    for(int ti = 0; ti < nt; ti++){
        model -> propagate(start, 
                            4, 
                            &u[ti],
                            1,
                            (int)(t[ti] / dt),
                            state,
                            dt);
    }
   
    for(int i = 0; i < 4; i++){
        cout << state[i] << ",";
    }
    cout << endl << "reference:"<< endl;
    for(int i = 0; i < 4; i++){
        cout << goal[i] << ",";
    }
    cout << endl;

    // vector<vector<double>> path = cem.rolling(start, goal);
    // for(int i = 0; i < path.size(); i++){
    //     for(int si = 0; si < 4; si++){
    //         cout<<path.at(i).at(si)<<",";
    //     }
    //     cout<<endl;
    // }

    return 0;
}