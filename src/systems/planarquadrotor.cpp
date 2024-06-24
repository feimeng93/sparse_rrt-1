/**
 * @file point.cpp
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

#include "systems/planarquadrotor.hpp"
#include "utilities/random.hpp"
#include "image_creation/svg_image.hpp"
#include <cmath>
#include <assert.h>

#define MIN_Z -2
#define MAX_Z 2
#define MIN_Y -2
#define MAX_Y 2
#define MIN_THETA -M_PI/3
#define MAX_THETA M_PI/3


#define MIN_V -1.0
#define MAX_V 1.0

#define MIN_CONTROL -19.62
#define MAX_CONTROL 19.62

#define m 2.0
#define g 9.81
#define I 1.0
#define l 0.2

#define STATE_Y 0
#define STATE_Z 1
#define STATE_THETA 2
#define STATE_VY 3
#define STATE_VZ 4
#define STATE_W 5


double planar_quadrotor_t::distance(const double* point1, const double* point2, unsigned int)
{
    double val = fabs(point1[STATE_THETA]-point2[STATE_THETA]);
    if(val > M_PI)
            val = 2*M_PI-val;
    return std::sqrt(val * val + pow(point1[STATE_Y]-point2[STATE_Y], 2.0) + pow(point1[STATE_Z]-point2[STATE_Z], 2.0)+ pow(point1[STATE_VY]-point2[STATE_VY], 2.0)+ pow(point1[STATE_VZ]-point2[STATE_VZ], 2.0)+ pow(point1[STATE_W]-point2[STATE_W], 2.0));
}
	
bool planar_quadrotor_t::propagate(
    const double* start_state, unsigned int state_dimension,
    const double* control, unsigned int control_dimension,
    int num_steps, double* result_state, double integration_step)
{
	temp_state[0] = start_state[0];
	temp_state[1] = start_state[1];
	temp_state[2] = start_state[2];
	temp_state[3] = start_state[3];
	temp_state[4] = start_state[4];
	temp_state[5] = start_state[5];
	bool validity = true;

	for(int i=0;i<num_steps;i++)
	{
		update_derivative(control);
		temp_state[0] += integration_step*deriv[0];
		temp_state[1] += integration_step*deriv[1];
		temp_state[2] += integration_step*deriv[2];
		temp_state[3] += integration_step*deriv[3];
		temp_state[4] += integration_step*deriv[4];
		temp_state[5] += integration_step*deriv[5];

		enforce_bounds();
		// validity = validity && valid_state();
		if (valid_state() == true){
				result_state[0] = temp_state[0];
				result_state[1] = temp_state[1];
				result_state[2] = temp_state[2];
				result_state[3] = temp_state[3];
				result_state[4] = temp_state[4];
				result_state[5] = temp_state[5];

			}
		else
		{
			// Found the earliest invalid position. break the loop and return
			validity = false; // need to update validity because one node is invalid, the propagation fails
			break;
		}
	}
	// result_state[0] = temp_state[0];
	// result_state[1] = temp_state[1];
	return validity;
}

void planar_quadrotor_t::enforce_bounds()
{
	if(temp_state[0]<MIN_Y)
		temp_state[0]=MIN_Y;
	else if(temp_state[0]>MAX_Y)
		temp_state[0]=MAX_Y;

	if(temp_state[1]<MIN_Z)
		temp_state[1]=MIN_Z;
	else if(temp_state[1]>MAX_Z)
		temp_state[1]=MAX_Z;

	if(temp_state[2]<-MIN_THETA)
		temp_state[2]=MIN_THETA;
	else if(temp_state[1]>MAX_THETA)
		temp_state[2]=MAX_THETA;
	
	if(temp_state[3]<-MIN_V)
		temp_state[3]=MIN_V;
	else if(temp_state[3]>MAX_V)
		temp_state[3]=MAX_V;

	if(temp_state[4]<-MIN_V)
		temp_state[4]=MIN_V;
	else if(temp_state[4]>MAX_V)
		temp_state[4]=MAX_V;

	if(temp_state[5]<-MIN_V)
		temp_state[5]=MIN_V;
	else if(temp_state[5]>MAX_V)
		temp_state[5]=MAX_V;
}


bool planar_quadrotor_t::valid_state()
{
	bool obstacle_collision = false;
	//any obstacles need to be checked here
	for(unsigned i=0;i<obs_min_max.size() && !obstacle_collision;i++)
	{
		if(	temp_state[0]>=obs_min_max[i][0] && 
			temp_state[0]<=obs_min_max[i][1] && 
			temp_state[1]>=obs_min_max[i][2] && 
			temp_state[1]<=obs_min_max[i][3])
		{
			obstacle_collision = true;
		}
	}

	return !obstacle_collision && 
			(temp_state[0]!=MIN_Y) &&
			(temp_state[0]!=MAX_Y) &&
			(temp_state[1]!=MIN_Z) &&
			(temp_state[1]!=MAX_Z);
}

void planar_quadrotor_t::update_derivative(const double* control)
{
    double _y = temp_state[0];
    double _z = temp_state[1];
	double _theta = temp_state[2];
	double _vy = temp_state[3];
	double _vz = temp_state[4];
	double _w = temp_state[5];

    deriv[0] = _vy;
    deriv[1] = _vz;
	deriv[2] = _w;
	deriv[3] = -(1/m)*sin(_theta)*(control[0] + control[1]);
	deriv[4] = (1/m)*cos(_theta)* (control[0] + control[1]) - g;
	deriv[5] = (1/I)*l*(control[1] - control[0]);
}

std::tuple<double, double> planar_quadrotor_t::visualize_point(const double* state, unsigned int state_dimension) const
{
	double y = (state[0]-MIN_Y)/(MAX_Y-MIN_Y);
	double z = (state[1]-MIN_Z)/(MAX_Z-MIN_Z);
	return std::make_tuple(y, z);
}

std::string planar_quadrotor_t::visualize_obstacles(int image_width, int image_height) const
{
    svg::Dimensions dims(image_width, image_height);
    svg::DocumentBody doc(svg::Layout(dims, svg::Layout::BottomLeft));
	double temp[2];
	for(unsigned i=0;i<obs_min_max.size();i++)
	{
		temp[0] = obs_min_max[i][0];
		temp[1] = obs_min_max[i][3];
		double x, y;
		std::tie(x, y) = this->visualize_point(temp, 2);

		doc<<svg::Rectangle(svg::Point(x*dims.width, y*dims.height),
							(obs_min_max[i][1]-obs_min_max[i][0])/(MAX_Y-MIN_Y) * dims.width,
							(obs_min_max[i][3]-obs_min_max[i][2])/(MAX_Z-MIN_Z) * dims.height,
							svg::Color::Blue);
	}
    return doc.toString();
}

std::vector<std::pair<double, double> > planar_quadrotor_t::get_state_bounds() const {
	return {
			{MIN_Y,MAX_Y},
			{MIN_Z,MAX_Z},
			{MIN_THETA,MAX_THETA},
			{MIN_V,MAX_V},
			{MIN_V,MAX_V},
			{MIN_V,MAX_V}
	};
}


std::vector<std::pair<double, double> > planar_quadrotor_t::get_control_bounds() const {
	return {
			{MIN_CONTROL, MAX_CONTROL},
			{MIN_CONTROL, MAX_CONTROL}
	};
}

std::vector<bool> planar_quadrotor_t::is_circular_topology() const{
    return {
            false,
            false,
			true,
			false,
			false,
			false
    };
}

