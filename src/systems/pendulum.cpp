/**
 * @file pendulum.cpp
 *
 * @copyright Software License Agreement (BSD License)
 * Original work Copyright (c) 2014, Rutgers the State University of New Jersey, New Brunswick
 * Modified work Copyright 2017 Oleg Y. Sinyavskiy
 * All Rights Reserved.
 * For a full description see the file named LICENSE.
 *
 * Original authors: Zakary Littlefield, Kostas Bekris
 * Modifications by: Oleg Y. Sinyavskiy
 * 
 */

#include "systems/pendulum.hpp"
#include "utilities/random.hpp"


#define _USE_MATH_DEFINES

#include <cmath>

#define MIN_W -8
#define MAX_W 8

#define MIN_TORQUE -8
#define MAX_TORQUE 8

#define LENGTH 1.0
#define MASS 1.0
#define DAMPING 1.0
#define g 9.8


bool pendulum_t::propagate(
    const double* start_state, unsigned int state_dimension,
    const double* control, unsigned int control_dimension,
    int num_steps, double* result_state, double integration_step)
{
	temp_state[0] = start_state[0]; 
	temp_state[1] = start_state[1];
	bool validity = true;
	for(int i=0;i<num_steps;i++)
	{
		double temp0 = temp_state[0];
		double temp1 = temp_state[1];
		temp_state[0] += integration_step*temp1;
		temp_state[1] += integration_step*(-g / LENGTH * sin(temp0)- DAMPING * LENGTH * temp1 / MASS + cos(temp0) * control[0] / (MASS * LENGTH));
		enforce_bounds();
		validity = validity && valid_state();
	}
	result_state[0] = temp_state[0];
	result_state[1] = temp_state[1];
	return validity;
}

void pendulum_t::enforce_bounds()
{
	if(temp_state[0]<-M_PI)
		temp_state[0]+=2*M_PI;
	else if(temp_state[0]>M_PI)
		temp_state[0]-=2*M_PI;

	if(temp_state[1]<MIN_W)
		temp_state[1]=MIN_W;
	else if(temp_state[1]>MAX_W)
		temp_state[1]=MAX_W;
}


bool pendulum_t::valid_state()
{
	return true;
}

std::tuple<double, double> pendulum_t::visualize_point(const double* state, unsigned int state_dimension) const
{
	double x = (state[0]+M_PI)/(2*M_PI);
	double y = (state[1]-MIN_W)/(MAX_W-MIN_W);
	return std::make_tuple(x, y);
}

std::vector<std::pair<double, double> > pendulum_t::get_state_bounds() const {
	return {
			{-M_PI,M_PI},
			{MIN_W,MAX_W},
	};
}


std::vector<std::pair<double, double> > pendulum_t::get_control_bounds() const {
	return {
			{MIN_TORQUE,MAX_TORQUE},
	};
}

std::vector<bool> pendulum_t::is_circular_topology() const {
	return {
            true,
			false
	};
}

double pendulum_t::distance(const double* point1, const double* point2, unsigned int)
{
    double val = fabs(point1[0]-point2[0]);
    if(val > M_PI)
            val = 2*M_PI-val;
    return std::sqrt(val * val + pow(point1[1]-point2[1], 2.0));

}