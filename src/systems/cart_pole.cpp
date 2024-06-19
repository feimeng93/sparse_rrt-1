/**
 * @file cart_pole.cpp
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


#include "systems/cart_pole.hpp"
#include "utilities/random.hpp"


#define _USE_MATH_DEFINES


#include <cmath>


#define I 10
#define L 0.5
#define M 1
#define m 0.1
#define g 9.8
// height of the cart
#define H 0.5

#define STATE_X 0
#define STATE_V 1
#define STATE_THETA 2
#define STATE_W 3
#define CONTROL_A 0

#define MIN_X -2.4
#define MAX_X 2.4
#define MIN_V -4000
#define MAX_V 4000
#define MAX_THETA 0.209
#define MIN_THETA -0.209
#define MIN_W -2000
#define MAX_W 2000


bool cart_pole_t::propagate(
    const double* start_state, unsigned int state_dimension,
    const double* control, unsigned int control_dimension,
    int num_steps, double* result_state, double integration_step)
{
        temp_state[0] = start_state[0]; 
        temp_state[1] = start_state[1];
        temp_state[2] = start_state[2];
        temp_state[3] = start_state[3];
        bool validity = false;
        
        for(int i=0;i<num_steps;i++)
        {
                update_derivative(control);
                temp_state[0] += integration_step*deriv[0];
                temp_state[1] += integration_step*deriv[1];
                temp_state[2] += integration_step*deriv[2];
                temp_state[3] += integration_step*deriv[3];
                enforce_bounds();
                //validity = validity && valid_state();
                if (valid_state() == true)
                {
                    result_state[0] = temp_state[0];
                    result_state[1] = temp_state[1];
                    result_state[2] = temp_state[2];
                    result_state[3] = temp_state[3];
                    validity = true;
                }
                else
                {
                    // Found the earliest invalid position. break the loop and return
                    validity = false; // need to update validity because one node is invalid, the propagation fails
                    break;
                }
        }        
        return validity;
}
//TODO confirm the theta bounds
void cart_pole_t::enforce_bounds()
{
        if(temp_state[0]<MIN_X)
                temp_state[0]=MIN_X;
        else if(temp_state[0]>MAX_X)
                temp_state[0]=MAX_X;


        // if(temp_state[2]<-M_PI)
        //         temp_state[2]+=2*M_PI;
        // else if(temp_state[2]>M_PI)
        //         temp_state[2]-=2*M_PI;
        if(temp_state[2]<MIN_THETA)
                temp_state[2]=MIN_THETA;
        else if(temp_state[2]>MAX_THETA)
                temp_state[2]=MAX_THETA;


}

bool cart_pole_t::valid_state()
{
        // check the pole with the rectangle to see if in collision
    // calculate the pole state
    // check if the position is within bound
    if (temp_state[0] < MIN_X or temp_state[0] > MAX_X)
    {
        return false;
    }
    double pole_x1 = temp_state[0];
    double pole_y1 = H;
    double pole_x2 = temp_state[0] + L * sin(temp_state[2]);
    double pole_y2 = H + L * cos(temp_state[2]);
    //std::cout << "state:" << temp_state[0] << "\n";
    //std::cout << "pole point 1: " << "(" << pole_x1 << ", " << pole_y1 << ")\n";
    //std::cout << "pole point 2: " << "(" << pole_x2 << ", " << pole_y2 << ")\n";
    for(unsigned int i = 0; i < obs_list.size(); i++)
    {
        // check if any obstacle has intersection with pole
        //std::cout << "obstacle " << i << "\n";
        //std::cout << "points: \n";
        for (unsigned int j = 0; j < 8; j += 2)
        {
            // check each line of the obstacle
            double x1 = obs_list[i][j];
            double y1 = obs_list[i][j+1];
            double x2 = obs_list[i][(j+2) % 8];
            double y2 = obs_list[i][(j+3) % 8];
            if (lineLine(pole_x1, pole_y1, pole_x2, pole_y2, x1, y1, x2, y2))
            {
                // intersect
                return false;
            }
        }
    }
    return true;
}

std::tuple<double, double> cart_pole_t::visualize_point(const double* state, unsigned int state_dimension) const
{
    double x = state[STATE_X] + (L / 2.0) * sin(state[STATE_THETA]);
    double y = -(L / 2.0) * cos(state[STATE_THETA]);

    x = (x-MIN_X)/(MAX_X-MIN_X);
    y = (y-MIN_X)/(MAX_X-MIN_X);
    return std::make_tuple(x, y);
}

void cart_pole_t::update_derivative(const double* control)
{
    double _v = temp_state[STATE_V];
    double _w = temp_state[STATE_W];
    double _theta = temp_state[STATE_THETA];
    double _a = control[CONTROL_A];
    double total_mass = M + m;

    deriv[STATE_X] = _v;
    deriv[STATE_THETA] = _w;
    double temp = (_a + L * m * _w  * _w * sin(_theta))/ total_mass;
    deriv[STATE_W] = (g * sin(_theta) - cos(_theta) * temp) / (L * (4.0 / 3.0 - m * cos(_theta) * cos(_theta) / total_mass));
    deriv[STATE_V] = temp - m * L * deriv[STATE_W] * cos(_theta) / total_mass;
}

bool cart_pole_t::lineLine(double x1, double y1, double x2, double y2, double x3, double y3, double x4, double y4)
// compute whether two lines intersect with each other
{
    // ref: http://www.jeffreythompson.org/collision-detection/line-rect.php
    // calculate the direction of the lines
    double uA = ((x4-x3)*(y1-y3) - (y4-y3)*(x1-x3)) / ((y4-y3)*(x2-x1) - (x4-x3)*(y2-y1));
    double uB = ((x2-x1)*(y1-y3) - (y2-y1)*(x1-x3)) / ((y4-y3)*(x2-x1) - (x4-x3)*(y2-y1));

    // if uA and uB are between 0-1, lines are colliding
    if (uA >= 0 && uA <= 1 && uB >= 0 && uB <= 1)
    {
        // intersect
        return true;
    }
    // not intersect
    return false;
}
std::vector<std::pair<double, double> > cart_pole_t::get_state_bounds() const {
    return {
            {MIN_X,MAX_X},
            {MIN_V,MAX_V},
            {-M_PI,M_PI},
            {MIN_W,MAX_W},
    };
}


std::vector<std::pair<double, double> > cart_pole_t::get_control_bounds() const {
    return {
            {-10,10},
    };
}


std::vector<bool> cart_pole_t::is_circular_topology() const {
    return {
            false,
            false,
            true,
            false
    };
}

double cart_pole_t::distance(const double* point1, const double* point2, unsigned int)
{
    double val = fabs(point1[STATE_THETA]-point2[STATE_THETA]);
    if(val > M_PI)
            val = 2*M_PI-val;
    return std::sqrt(val * val + pow(point1[0]-point2[0], 2.0) + pow(point1[1]-point2[1], 2.0)+ pow(point1[3]-point2[3], 2.0));
}