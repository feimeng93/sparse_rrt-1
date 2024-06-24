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

#include "systems/l.hpp"
#include "utilities/random.hpp"
#include "image_creation/svg_image.hpp"
#include <cmath>
#include <assert.h>

#define MIN_X -3
#define MAX_X 1
#define MIN_Y -3
#define MAX_Y 4

#define MIN_V -0.5
#define MAX_V 0.5


double l_t::distance(const double* point1, const double* point2, unsigned int state_dimension)
{
	return std::sqrt(pow(point1[0]-point2[0], 2.0) + pow(point1[1]-point2[1], 2.0));
}
	
bool l_t::propagate(
    const double* start_state, unsigned int state_dimension,
    const double* control, unsigned int control_dimension,
    int num_steps, double* result_state, double integration_step)
{
	temp_state[0] = start_state[0];
	temp_state[1] = start_state[1];
	bool validity = true;

	for(int i=0;i<num_steps;i++)
	{
		update_derivative(control);
		temp_state[0] += integration_step*deriv[0];
		temp_state[1] += integration_step*deriv[1];
		enforce_bounds();
		// validity = validity && valid_state();
		if (valid_state() == true){
				result_state[0] = temp_state[0];
				result_state[1] = temp_state[1];
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

void l_t::enforce_bounds()
{
	if(temp_state[0]<MIN_X)
		temp_state[0]=MIN_X;
	else if(temp_state[0]>MAX_X)
		temp_state[0]=MAX_X;

	if(temp_state[1]<MIN_Y)
		temp_state[1]=MIN_Y;
	else if(temp_state[1]>MAX_Y)
		temp_state[1]=MAX_Y;
}


bool l_t::valid_state()
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
			(temp_state[0]!=MIN_X) &&
			(temp_state[0]!=MAX_X) &&
			(temp_state[1]!=MIN_Y) &&
			(temp_state[1]!=MAX_Y);
}

void l_t::update_derivative(const double* control)
{
    double _x1 = temp_state[0];
    double _x2 = temp_state[1];

    deriv[0] = 0.5*_x2;
    deriv[1] = -0.1*_x1+0.2*_x2+ control[0];
}

std::tuple<double, double> l_t::visualize_point(const double* state, unsigned int state_dimension) const
{
	double x = (state[0]-MIN_X)/(MAX_X-MIN_X);
	double y = (state[1]-MIN_Y)/(MAX_Y-MIN_Y);
	return std::make_tuple(x, y);
}

std::string l_t::visualize_obstacles(int image_width, int image_height) const
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
							(obs_min_max[i][1]-obs_min_max[i][0])/(MAX_X-MIN_X) * dims.width,
							(obs_min_max[i][3]-obs_min_max[i][2])/(MAX_Y-MIN_Y) * dims.height,
							svg::Color::Blue);
	}
    return doc.toString();
}

std::vector<std::pair<double, double> > l_t::get_state_bounds() const {
	return {
			{MIN_X,MAX_X},
			{MIN_Y,MAX_Y}
	};
}


std::vector<std::pair<double, double> > l_t::get_control_bounds() const {
	return {
			{MIN_V, MAX_V}
	};
}

std::vector<bool> l_t::is_circular_topology() const{
    return {
            false,
            false
    };
}

