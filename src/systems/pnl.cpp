/**
 * @file pnl.cpp
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


#include "systems/pnl.hpp"
#include "image_creation/svg_image.hpp"
#include <cmath>
#include <assert.h>

#define MIN_X  -0.5
#define MIN_Y  -0.5
#define MIN_Z  -2

#define MAX_X  1.5
#define MAX_Y  0.6
#define MAX_Z  1

#define MIN_V -0.2
#define MAX_V 0.2

double pnl_t::distance(const double* point1, const double* point2, unsigned int state_dimension)
{
	return std::sqrt(pow(point1[0]-point2[0], 2.0) + pow(point1[1]-point2[1], 2.0)+ pow(point1[2]-point2[2], 2.0));
}

bool pnl_t::propagate(
		const double* start_state, unsigned int state_dimension,
        const double* control, unsigned int control_dimension,
	    int num_steps, double* result_state, double integration_step){
    for(int si = 0; si < state_dimension; si++){
        temp_state[si] = start_state[si];
    }
    validity = true;
    for(int t = 0; t < num_steps; t++)
    {
        update_derivative(control);
        for(int si = 0; si < state_dimension; si++){
            temp_state[si] += deriv[si] * integration_step;
        }
        enforce_bounds();
        validity = validity && valid_state();
        if(validity){
            for(int si = 0; si < state_dimension; si++){
                result_state[si] = temp_state[si];
            }
        } else {
            break;
        }
    }
    return validity;
}

void pnl_t::enforce_bounds()
{
	if(temp_state[0]<MIN_X)
		temp_state[0]=MIN_X;
	else if(temp_state[0]>MAX_X)
		temp_state[0]=MAX_X;

	if(temp_state[1]<MIN_Y)
		temp_state[1]=MIN_Y;
	else if(temp_state[1]>MAX_Y)
		temp_state[1]=MAX_Y;

	if(temp_state[2]<MIN_Z)
		temp_state[2]=MIN_Z;
	else if(temp_state[1]>MAX_Z)
		temp_state[2]=MAX_Z;
}


bool pnl_t::valid_state()
{
	bool obstacle_collision = false;
	//any obstacles need to be checked here
	for(unsigned i=0;i<obs_min_max.size() && !obstacle_collision;i++)
	{
		if(	temp_state[0]>=obs_min_max.at(i).at(0) && 
			temp_state[0]<=obs_min_max.at(i).at(1) && 
			temp_state[1]>=obs_min_max.at(i).at(2) && 
			temp_state[1]<=obs_min_max.at(i).at(3) &&
			temp_state[2]>=obs_min_max.at(i).at(4) &&
			temp_state[2]<=obs_min_max.at(i).at(5))
		{
			obstacle_collision = true;
		}
	}

	return !obstacle_collision && 
			(temp_state[0]!=MIN_X) &&
			(temp_state[0]!=MAX_X) &&
			(temp_state[1]!=MIN_Y) &&
			(temp_state[1]!=MAX_Y) &&
			(temp_state[2]!=MIN_Z) &&
			(temp_state[2]!=MAX_Z);
}

void pnl_t::update_derivative(const double* control)
{
    double _x1 = temp_state[0];
    double _x2 = temp_state[1];
	double _x3 = temp_state[2];

    deriv[0] = _x3;
    deriv[1] = -_x1 + std::pow(_x1, 3.0)/6 - _x3;
	deriv[2] = control[0];
}

std::tuple<double, double> pnl_t::visualize_point(const double* state, unsigned int state_dimension) const
{
	 return {};
}


std::tuple<double, double, double> pnl_t::visualize_3Dpoint(const double* state, unsigned int state_dimension) const
{
	double x = (state[0]-MIN_X)/(MAX_X-MIN_X);
	double y = (state[1]-MIN_Y)/(MAX_Y-MIN_Y);
	double z = (state[2]-MIN_Z)/(MAX_Z-MIN_Z);
	return std::make_tuple(x, y, z);
}

std::string pnl_t::visualize_obstacles(int image_width, int image_height) const
{
    svg::Dimensions dims(image_width, image_height);
    svg::DocumentBody doc(svg::Layout(dims, svg::Layout::BottomLeft));
	double temp[3];
	for(unsigned i=0;i<obs_min_max.size();i++)
	{
		// temp[0] = obs_list[i][0];
		// temp[1] = obs_list[i][1];
		// temp[2] = obs_list[i][2];
		double x, y;
		std::tie(x, y) = this->visualize_point(temp, 3);

		doc<<svg::Rectangle(svg::Point(x*dims.width, y*dims.height), //TODO: fix this in 3D
							(obs_min_max.at(i).at(1)-obs_min_max.at(i).at(0))/(MAX_X-MIN_X) * dims.width,
							(obs_min_max.at(i).at(3)-obs_min_max.at(i).at(4))/(MAX_Y-MIN_Y) * dims.height,
							svg::Color::Red);
	}
    return doc.toString();
}

std::vector<std::pair<double, double> > pnl_t::get_state_bounds() const {
	return {
			{MIN_X,MAX_X},
			{MIN_Y,MAX_Y},
			{MIN_Z,MAX_Z}
	};
}


std::vector<std::pair<double, double> > pnl_t::get_control_bounds() const {
	return {
			{MIN_V, MAX_V}
	};
}

std::vector<bool> pnl_t::is_circular_topology() const{
    return {
            false,
            false,
			false
    };
}

// void pnl_t::normalize(const double* state, double* normalized){
// }

// void pnl_t::denormalize(double* normalized,  double* state){


// }

// double pnl_t::get_loss(double* point1, const double* point2, double* weight){
// }
