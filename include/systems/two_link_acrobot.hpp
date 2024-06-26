/**
 * @file two_link_acrobot.hpp
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

#ifndef SPARSE_TWO_LINK_ACROBOT_HPP
#define SPARSE_TWO_LINK_ACROBOT_HPP


#include "systems/system.hpp"
#include <Eigen/Dense>
#include <iostream>
#include <vector>
#include <tuple>
#include <map>
using Eigen::Matrix2d;
using Eigen::Vector2d;
using Eigen::Vector4d;

class two_link_acrobot_t : public system_t
{
public:
	two_link_acrobot_t(){
		state_dimension = 4;
		control_dimension = 1;
		temp_state = new double[state_dimension];
		deriv = new double[state_dimension];
	}
	two_link_acrobot_t(std::vector<std::vector<double>>& _obs_list, double width)
	{
		state_dimension = 4;
		control_dimension = 1;
		temp_state = new double[state_dimension];
		deriv = new double[state_dimension];
		// copy the items from _obs_list to obs_list
		for(unsigned i=0; i<_obs_list.size(); i++)
		{
			// each obstacle is represented by its middle point
			std::vector<double> obs(4*2);
			// calculate the four points representing the rectangle in the order
			// UL, UR, LR, LL
			// the obstacle points are concatenated for efficient calculation
			double x = _obs_list[i][0];
			double y = _obs_list[i][1];
			obs[0] = x - width / 2;  obs[1] = y + width / 2;
			obs[2] = x + width / 2;  obs[3] = y + width / 2;
			obs[4] = x + width / 2;  obs[5] = y - width / 2;
			obs[6] = x - width / 2;  obs[7] = y - width / 2;
			obs_list.push_back(obs);
		}
	
	}
	virtual ~two_link_acrobot_t()
	{
		delete temp_state;
	    delete deriv;
		// clear the vector
		obs_list.clear();
	}

	static Vector4d dynamics(const Vector4d &y, const Vector2d &u);

	static void rk4_step(Vector4d &y, const Vector2d &u, const double integration_step);

	/**
	 * @copydoc system_t::distance(double*, double*)
	 */
	static double distance(const double* point1, const double* point2, unsigned int);

	/**
	 * @copydoc system_t::propagate(double*, double*, int, int, double*, double& )
	 */
	virtual bool propagate(
		const double* start_state, unsigned int state_dimension,
        const double* control, unsigned int control_dimension,
	    int num_steps, double* result_state, double integration_step);

	/**
	 * @copydoc system_t::enforce_bounds()
	 */
	virtual void enforce_bounds();
	
	/**
	 * @copydoc system_t::valid_state()
	 */
	virtual bool valid_state();

	/**
	 * @copydoc system_t::visualize_point(double*, svg::Dimensions)
	 */
	std::tuple<double, double> visualize_point(const double* state, unsigned int state_dimension) const;

	/**
	 * @copydoc system_t::get_state_bounds()
	 */
    std::vector<std::pair<double, double>> get_state_bounds() const override;

    /**
	 * @copydoc system_t::get_control_bounds()
	 */
    std::vector<std::pair<double, double>> get_control_bounds() const override;

    /**
	 * @copydoc system_t::is_circular_topology()
	 */
    std::vector<bool> is_circular_topology() const override;
/**
	 * compute the error between to angles from -pi to pi and wrap the error back to 0~pi
	 */
	std::vector<std::vector<double>> obs_list;

	
protected:
	double* deriv;
	void update_derivative(const double* control);
	bool lineLine(double x1, double y1, double x2, double y2, double x3, double y3, double x4, double y4);
};


#endif