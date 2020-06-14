#include "systems/quadrotor.hpp"
#include <iostream>

#define MIN_X -5
#define MAX_X 5
#define MIN_Q -1
#define MAX_Q 1

#define MIN_V -1.
#define MAX_V 1.

#define MASS_INV 1.
#define BETA 1.
#define EPS 2.107342e-08
#define g 9.81

#define MIN_C1 -15
#define MAX_C1 0.
#define MIN_C -1
#define MAX_C 1.

void quadrotor_t::enforce_bounds_SO3(double* qstate){
    double nrmsq = qstate[0]*qstate[0] + qstate[1]*qstate[1] + qstate[2]*qstate[2] + qstate[3]*qstate[3];
    double error = std::abs(1.0 - nrmsq);
    if (error < EPS) {
        double scale = 2.0 / (1.0 + nrmsq);
        qstate[0] *= scale;
        qstate[1] *= scale;
        qstate[2] *= scale;
        qstate[3] *= scale;
    } else {
        if (nrmsq < 1e-6){
            qstate = new double[4]();
            qstate[3] = 1;
        } else {
            double scale = 1.0 / std::sqrt(nrmsq);
            qstate[0] *= scale;
            qstate[1] *= scale;
            qstate[2] *= scale;
            qstate[3] *= scale;
        }
    }
}


bool quadrotor_t::propagate(
		const double* start_state, unsigned int state_dimension,
        const double* control, unsigned int control_dimension,
	    int num_steps, double* result_state, double integration_step){
    for(int i_s = 0; i_s < state_dimension; i_s++){
        temp_state[i_s] = start_state[i_s];

    }
    for(int t = 0; t < num_steps; t++)
    {
        update_derivative(control);
        for(int i_s = 0; i_s < state_dimension; i_s++){
            temp_state[i_s] += deriv[i_s] * integration_step;
        }
        enforce_bounds();
        // validity = validity && true; //valid_state();
    }
    for(int i_s = 0; i_s < state_dimension; i_s++){
        result_state[i_s] = temp_state[i_s];
    }
    return true;//validity;
}

bool quadrotor_t::valid_state(){
    return true;
}

void quadrotor_t::enforce_bounds(){
    // for v and w
    for(int i_s = 7; i_s < state_dimension; i_s++)
    if(temp_state[i_s] < MIN_V){
        temp_state[i_s] = MIN_V;

    }else if(temp_state[i_s] > MAX_V){
        temp_state[i_s] = MAX_V;
    }
    enforce_bounds_SO3(temp_state+3);

};

double quadrotor_t::distance(double* point1, double* point2){
    double dis = 0;
    for(int i_s = 0; i_s < state_dimension; i_s++){
        dis += (point1[i_s] - point2[i_s]) * (point1[i_s] - point2[i_s]);
    }
    return dis;
};		

void quadrotor_t::update_derivative(const double* control){
    double* u = new double[control_dimension];
    // enforce control
    if(control[0] > MAX_C1){
            u[0] = MAX_C1;
        } else if (u[0] < MIN_C1){
            u[0] = MIN_C1;
        } else {
            u[0] = control[0];
        }
        for(int i_u = 1; i_u < control_dimension; i_u++){
            if(control[i_u] > MAX_C){
            u[i_u] = MAX_C;
            } else if (control[i_u] < MIN_C){
                u[i_u] = MIN_C;
            } else {
                u[i_u] = control[i_u];
            }
        }
    // dx/dt = v
    deriv[0] = temp_state[7];
    deriv[1] = temp_state[8];
    deriv[2] = temp_state[9];
    double *qomega = new double[4];
    qomega[0] = .5 * temp_state[10];
    qomega[1] = .5 * temp_state[11];
    qomega[2] = .5 * temp_state[12];
    qomega[3] = 0;
    enforce_bounds_SO3(qomega);
    double delta = temp_state[3] * qomega[0] + temp_state[4] * qomega[1] + temp_state[5] * qomega[2];
    deriv[3] = qomega[0] - delta * temp_state[3];
    deriv[4] = qomega[1] - delta * temp_state[4];
    deriv[5] = qomega[2] - delta * temp_state[5];
    deriv[6] = qomega[3] - delta * temp_state[6];
    deriv[7] = MASS_INV * (-2*u[0]*(temp_state[6]*temp_state[4] + temp_state[3]*temp_state[5]) - BETA * temp_state[7]);
    deriv[8] = MASS_INV * (-2*u[0]*(temp_state[4]*temp_state[5] - temp_state[6]*temp_state[3]) - BETA * temp_state[8]);
    deriv[9] = MASS_INV * (-u[0]*(temp_state[6]*temp_state[6]-temp_state[3]*temp_state[3]-temp_state[4]*temp_state[4]+temp_state[5]*temp_state[5]) - BETA * temp_state[9]) - 9.81;

    deriv[10] = u[1];
    deriv[11] = u[2];
    deriv[12] = u[3];

};

std::vector<std::pair<double, double> > quadrotor_t::get_control_bounds() const{
    return {
            {MIN_C1, MAX_C1},
            {MIN_C, MAX_C},
            {MIN_C, MAX_C},
            {MIN_C, MAX_C},
    };
}

std::vector<std::pair<double, double> > quadrotor_t::get_state_bounds() const {
    return {
            {MIN_X, MAX_X},
            {MIN_X, MAX_X},
            {MIN_X, MAX_X},

            {MIN_Q, MAX_Q},
            {MIN_Q, MAX_Q},
            {MIN_Q, MAX_Q},
            {MIN_Q, MAX_Q},

            {MIN_V,MAX_V},
            {MIN_V,MAX_V},
            {MIN_V,MAX_V},

            {MIN_V,MAX_V},
            {MIN_V,MAX_V},
            {MIN_V,MAX_V}
    };
}

std::vector<bool> quadrotor_t::is_circular_topology() const{
    return {
            false,
            false,
            false,
            
            false,
            false,
            false,
            false,
            
            false,
            false,
            false,

            false,
            false,
            false
    };
}

void quadrotor_t::normalize(const double* state, double* normalized){
    for(int i = 0; i < state_dimension; i++){
        normalized[i] = state[i];
    }

}
void quadrotor_t::denormalize(double* normalized,  double* state){
    for(int i = 0; i < state_dimension; i++){
        state[i] = normalized[i]; 
    }
}

std::tuple<double, double> quadrotor_t::visualize_point(const double* state, unsigned int state_dimension) const{
    return std::make_tuple(0, 0);
}
