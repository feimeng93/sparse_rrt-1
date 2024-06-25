/**
 * @file distance_functions.cpp
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

#include <assert.h>
#include "systems/distance_functions.h"
#include "systems/two_link_acrobot.hpp"
#include "systems/planarquadrotor.hpp"
#include "systems/cart_pole.hpp"
#include "systems/pendulum.hpp"
#include "systems/l.hpp"
#include "systems/pnl.hpp"

double two_link_acrobot_distance::distance(const double* p0, const double* p1, unsigned int state_dimensions) const
{
    return two_link_acrobot_t::distance(p0, p1, state_dimensions);
}

double planarquad_distance::distance(const double* p0, const double* p1, unsigned int state_dimensions) const
{
    return planar_quadrotor_t::distance(p0, p1, state_dimensions);

}

double pendulum_distance::distance(const double* p0, const double* p1, unsigned int state_dimensions) const
{
    return pendulum_t::distance(p0, p1, state_dimensions);

}

double cart_pole_distance::distance(const double* p0, const double* p1, unsigned int state_dimensions) const
{
    return cart_pole_t::distance(p0, p1, state_dimensions);
}

double l_distance::distance(const double* p0, const double* p1, unsigned int state_dimensions) const
{
    return l_t::distance(p0, p1, state_dimensions);
}

double pnl_distance::distance(const double* p0, const double* p1, unsigned int state_dimensions) const
{
    return pnl_t::distance(p0, p1, state_dimensions);
}