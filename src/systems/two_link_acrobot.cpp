/**
 * @file two_link_acrobot.cpp
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


#include "systems/two_link_acrobot.hpp"


#define _USE_MATH_DEFINES

using Eigen::Matrix2d;
using Eigen::Vector2d;
using Eigen::Vector4d;

#define l1 1.0
#define l2 1.0
#define m1 1.0
#define m2 1.0
#define g  9.8
// #define integration_step 0.01

#define LENGTH 1.0
// #define lc  .5
// #define lc2  .25
// #define I1  0.2
// #define I2  1.0



#define STATE_THETA_1 0
#define STATE_THETA_2 1
#define STATE_V_1 2
#define STATE_V_2 3
#define CONTROL_T 0

#define MIN_V_1 -8
#define MAX_V_1 8
#define MIN_V_2 -8
#define MAX_V_2 8
#define MIN_T -6
#define MAX_T 6

using Eigen::Matrix2d;
using Eigen::Vector2d;
using Eigen::Vector4d;


double two_link_acrobot_t::distance(const double *point1, const double *point2, unsigned int state_dimension)
{
        double x = (LENGTH) * cos(point1[STATE_THETA_1] - M_PI / 2)+(LENGTH) * cos(point1[STATE_THETA_1] + point1[STATE_THETA_2] - M_PI / 2);
        double y = (LENGTH) * sin(point1[STATE_THETA_1] - M_PI / 2)+(LENGTH) * sin(point1[STATE_THETA_1] + point1[STATE_THETA_2] - M_PI / 2);
        double x2 = (LENGTH) * cos(point2[STATE_THETA_1] - M_PI / 2)+(LENGTH) * cos(point2[STATE_THETA_1] + point2[STATE_THETA_2] - M_PI / 2);
        double y2 = (LENGTH) * sin(point2[STATE_THETA_1] - M_PI / 2)+(LENGTH) * sin(point2[STATE_THETA_1] + point2[STATE_THETA_2] - M_PI / 2);
        return std::sqrt(pow(x-x2,2.0)+pow(y-y2,2.0));
}

bool two_link_acrobot_t::propagate(
    const double* start_state, unsigned int state_dimension,
    const double* control, unsigned int control_dimension,
    int num_steps, double* result_state, double integration_step)
{
        Vector4d y;
        Vector2d u;
        y(0) = start_state[0];
        y(1) = start_state[1];
        y(2) = start_state[2];
        y(3) = start_state[3];
        u(0)= control[0];
        u(1)= control[1];
        bool validity = true;
        // find the last valid position, if no valid position is found, then return false
        for(int i=0;i<num_steps;i++)
        {        
            rk4_step(y, u, integration_step);
            //     update_derivative(control);
            temp_state[0] = y(0);
            temp_state[1] = y(1);
            temp_state[2] = y(2);
            temp_state[3] = y(3);
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
            validity = false;
            // Found the earliest invalid position. break the loop and return
            break;
            }
        }
        //result_state[0] = temp_state[0];
        //result_state[1] = temp_state[1];
        //result_state[2] = temp_state[2];
        //result_state[3] = temp_state[3];
        return validity;
    }

void two_link_acrobot_t::enforce_bounds()
{

    if(temp_state[0]<-M_PI)
            temp_state[0]+=2*M_PI;
    else if(temp_state[0]>M_PI)
            temp_state[0]-=2*M_PI;
    if(temp_state[1]<-M_PI)
            temp_state[1]+=2*M_PI;
    else if(temp_state[1]>M_PI)
            temp_state[1]-=2*M_PI;
    if(temp_state[2]<MIN_V_1)
            temp_state[2]=MIN_V_1;
    else if(temp_state[2]>MAX_V_1)
            temp_state[2]=MAX_V_1;
    if(temp_state[3]<MIN_V_2)
            temp_state[3]=MIN_V_2;
    else if(temp_state[3]>MAX_V_2)
            temp_state[3]=MAX_V_2;
}


bool two_link_acrobot_t::valid_state()
{
    // check the pole with the rectangle to see if in collision
    // calculate the pole state
    double pole_x0 = 0.;
    double pole_y0 = 0.;
    double pole_x1 = (LENGTH) * cos(temp_state[STATE_THETA_1] - M_PI / 2);
    double pole_y1 = (LENGTH) * sin(temp_state[STATE_THETA_1] - M_PI / 2);
    double pole_x2 = pole_x1 + (LENGTH) * cos(temp_state[STATE_THETA_1] + temp_state[STATE_THETA_2] - M_PI / 2);
    double pole_y2 = pole_y1 + (LENGTH) * sin(temp_state[STATE_THETA_1] + temp_state[STATE_THETA_2] - M_PI / 2);

    //std::cout << "state:" << temp_state[0] << "\n";
    //std::cout << "pole point 1: " << "(" << pole_x1 << ", " << pole_y1 << ")\n";
    //std::cout << "pole point 2: " << "(" << pole_x2 << ", " << pole_y2 << ")\n";
    for(unsigned int i = 0; i < obs_list.size(); i++)
    {
        // check if any obstacle has intersection with pole
        //std::cout << "obstacle " << i << "\n";
        //std::cout << "points: \n";
        for (unsigned int j = 0; j < 8; j+=2)
        {
            // check each line of the obstacle
            double x1 = obs_list[i][j];
            double y1 = obs_list[i][j+1];
            double x2 = obs_list[i][(j+2) % 8];
            double y2 = obs_list[i][(j+3) % 8];
            if (lineLine(pole_x0, pole_y0, pole_x1, pole_y1, x1, y1, x2, y2))
            {
                // intersect
                return false;
            }
            if (lineLine(pole_x1, pole_y1, pole_x2, pole_y2, x1, y1, x2, y2))
            {
                // intersect
                return false;
            }
        }
    }
    return true;
}

std::tuple<double, double> two_link_acrobot_t::visualize_point(const double* state, unsigned int state_dimension) const
{
    double x = (LENGTH) * cos(state[STATE_THETA_1] - M_PI / 2)+(LENGTH) * cos(state[STATE_THETA_1] + state[STATE_THETA_2] - M_PI / 2);
    double y = (LENGTH) * sin(state[STATE_THETA_1] - M_PI / 2)+(LENGTH) * sin(state[STATE_THETA_1] + state[STATE_THETA_2] - M_PI / 2);
    x = (x+2*LENGTH)/(4*LENGTH);
    y = (y+2*LENGTH)/(4*LENGTH);
    return std::make_tuple(x, y);
}


Vector4d two_link_acrobot_t::dynamics(const Vector4d &y, const Vector2d &u) {
    double th1 = y(0);
    double th2 = y(1);
    double dth1 = y(2);
    double dth2 = y(3);

    Vector4d f;
    Matrix2d M;
    Vector2d C;
    Vector2d G;

    double c2 = cos(th2);
    double s2 = sin(th2);

    M(0, 0) = m1 * pow(l1, 2) + m2 * (pow(l1, 2) + 2 * l1 * l2 * c2 + pow(l2, 2));
    M(0, 1) = m2 * (l1 * l2 * c2 + pow(l2, 2));
    M(1, 0) = M(0, 1);
    M(1, 1) = m2 * pow(l2, 2);

    C(0) = -m2 * l1 * l2 * s2 * (2 * dth1 * dth2 + pow(dth2, 2));
    C(1) = m2 * l1 * l2 * pow(dth1, 2) * s2;

    G(0) = (m1 + m2) * l1 * g * cos(th1) + m2 * g * l2 * cos(th1 + th2);
    G(1) = m2 * g * l2 * cos(th1 + th2);

    Vector2d ddth = M.inverse() * (u - C - G);

    f(0) = dth1;
    f(1) = dth2;
    f(2) = ddth(0);
    f(3) = ddth(1);

    return f;
}

void two_link_acrobot_t::rk4_step(Vector4d &y, const Vector2d &u, const double integration_step) {
    auto k1 = dynamics(y, u);
    auto k2 = dynamics(y + integration_step / 2 * k1, u);
    auto k3 = dynamics(y + integration_step / 2 * k2, u);
    auto k4 = dynamics(y + integration_step * k3, u);

    y += integration_step / 6 * (k1 + 2 * k2 + 2 * k3 + k4);
}


bool two_link_acrobot_t::lineLine(double x1, double y1, double x2, double y2, double x3, double y3, double x4, double y4)
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
std::vector<std::pair<double, double> > two_link_acrobot_t::get_state_bounds() const {
    return {
            {-M_PI,M_PI},
            {-M_PI,M_PI},
            {MIN_V_1,MAX_V_1},
            {MIN_V_2,MAX_V_2},
    };
}

std::vector<std::pair<double, double> > two_link_acrobot_t::get_control_bounds() const{
    return {
            {MIN_T,MAX_T},
            {MIN_T,MAX_T}
    };
}


std::vector<bool> two_link_acrobot_t::is_circular_topology() const{
    return {
            true,
            true,
            false,
            false
    };
}
