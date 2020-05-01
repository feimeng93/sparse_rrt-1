#include "systems/quadrotor.hpp"
#include <iostream>

#define MIN_C1 -15
#define MAX_C1 0.;
#define MIN_C -1
#define MAX_C 1.;
#define MIN_V -1.
#define MAX_V 1.;
#define MASS_INV 1.;
#define BETA 1.;
#define EPS 2.107342e-08;
#define g 9.81;


void quadrotor_t::enforce_bounds_SO3(double* qstate){
    double nrmsq = qstate[0]*qstate[0] + qstate[1]*qstate[1] + qstate[2]*qstate[2] + qstate[3]*qstate[3];
    double error = std::abs(1.0 - nrmsq);
    if (error < EPS){
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


bool quadrotor_t::ropagate(
		const double* start_state, unsigned int state_dimension,
        const double* control, unsigned int control_dimension,
	    int num_steps, double* result_state, double integration_step){
    for(int i_s = 0; i_s < state_dimension; i_s++){
        q[i_s] = start_state[i_s];

    }
    for(int t = 0; t < num_steps; t++)
    {
        update_derivative(control);
        for(int i_s = 0; i_s < state_dimension; i_s++){
            q[i_s] += qdot[i_s] * integration_step;
        }
        enforce_bounds();
        // validity = validity && true; //valid_state();
    }
    for(int i_s = 0; i_s < STATE_DIM; i_s++){
        result_state[i_s] = q[i_s];
    }
    return validity;
}

bool quadrotor_t::valid_state(){
    return true;
}

void quadrotor_t::enforce_bounds(){
    // for v and w
    for(int i_s = 7; i_s < STATE_DIM; i_s++)
    if(q[i_s] < MIN_V){
        q[i_s] = MIN_V;

    }else if(q[i_s] > MAX_V){
        q[i_s] = MAX_V;
    }
    enforce_bounds_SO3(q+3);

};

double quadrotor_t::distance(double* point1, double* point2){
    double dis = 0;
    for(int i_s = 0; i_s < STATE_DIM; i_s++){
        dis += (point1[i_s] - point2[i_s]) * (point1[i_s] - point2[i_s]);
    }
    return dis;
};		

void quadrotor_t::update_derivative(double* control){
    // enforce control
    if(control[0] > MAX_C1){
        control[0] = MAX_C1;
    } else if (control[0] < MIN_C1){
        control[0] = MIN_C1;
    }
    for(int i_u = 1; i_u < control_dimension; i_u++){
        if(control[i_u] > MAX_C){
        control[i_u] = MAX_C;
        } else if (control[i_u] < MIN_C){
            control[i_u] = MIN_C;
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
    deriv[7] = MASS_INV * (-2*control[0]*(temp_state[6]*temp_state[4] + temp_state[3]*temp_state[5]) - BETA * temp_state[7]);
    deriv[8] = MASS_INV * (-2*control[0]*(temp_state[4]*temp_state[5] - temp_state[6]*temp_state[3]) - BETA * temp_state[8]);
    deriv[9] = MASS_INV * (-control[0]*(temp_state[6]*temp_state[6]-temp_state[3]*temp_state[3]-temp_state[4]*temp_state[4]+temp_state[5]*temp_state[5]) - BETA * temp_state[9]) - 9.81;

    deriv[10] = control[1];
    deriv[11] = control[2];
    deriv[12] = control[3];

};

