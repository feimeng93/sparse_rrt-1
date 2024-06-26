/**
 * @file cart_pole_obs.hpp
 *
 * @copyright Software License Agreement (BSD License)
 * Original work Copyright (c) 2014, Rutgers the State University of New Jersey, New Brunswick
 * Modified work Copyright 2017 Oleg Y. Sinyavskiy
 * All Rights Reserved.
 * For a full description see the file named LICENSE.
 *
 * Original authors: Zakary Littlefield, Kostas Bekris
 * Modifications by: Yinglong Miao
 *
 */

#ifndef SPARSE_CART_POLE_OBS_HPP
#define SPARSE_CART_POLE_OBS_HPP


#include "systems/system.hpp"
#include "utilities/random.hpp"
#include "image_creation/svg_image.hpp"

#include <iostream>

#define _USE_MATH_DEFINES


#include <cmath>

class cart_pole_t : public system_t
{
public:
	cart_pole_t(std::vector<std::vector<double>>& _obs_list, double width)
	/**
	* :obs_list: list of obstacles represented by 2d array representing the middle point
	* :width: the width of the obstacle
	*/
	{
		state_dimension = 4;
		control_dimension = 1;
		temp_state = new double[state_dimension];
		deriv = new double[state_dimension];
    	double pointSpacing = 0.1;

		// copy the items from _obs_list to obs_list
		for(unsigned i=0;i<_obs_list.size();i++)
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

			// obs_list_points_gen(obs_list);

			// Compute the number of points to generate based on desired density.
			// int numPointsX = static_cast<int>((obs[2] - obs[0]) / pointSpacing);
			// int numPointsY = static_cast<int>((obs[1] - obs[5]) / pointSpacing);
			// std::vector<double>obs_point(2);
			// std::vector<std::vector<double>> obs_points(numPointsX*numPointsY, std::vector<double>(2));
			// std::random_device rd;
			// std::mt19937 gen(rd());
			// std::uniform_real_distribution<> disX(obs[0], obs[2]);
			// std::uniform_real_distribution<> disY(obs[5], obs[1]);
			// for(int i = 0; i < numPointsX; ++i)
			// {
			// 	for(int j = 0; j < numPointsY; ++j)
			// 	{
			// 		obs_point[0] = disX(gen);
			// 		obs_point[1] = disY(gen);
			// 		obs_points.push_back(obs_point);
			// 	}
			// }
			// obs_list_points.push_back(obs_points);
		}
	}
	virtual ~cart_pole_t(){
	    delete temp_state;
	    delete deriv;
		// clear the vector
		obs_list.clear();
	}

	/**
	 * @copydoc system_t::propagate(const double*, const double*, int, int, double*, double& )
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
	std::tuple<double, double> visualize_point(const double* state, unsigned int state_dimension) const override;

	std::string visualize_obstacles(int image_width, int image_height) const;

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

	// double angular_error(double angle, double goal);
	
	// double get_loss(double* state, const double* goal, double* weight);

	// /**
	//  * normalize state to [-1,1]^4
	//  */
	// void normalize(const double* state, double* normalized);

	// /**
	//  * denormalize state back 
	//  */ 
	// void denormalize(double* normalized,  double* state);

	static double distance(const double* point1, const double* point2, unsigned int);
	// static double state_distance(const double* point1, const double* point2, unsigned int);


protected:
	double* deriv;
	void update_derivative(const double* control);
	// for obstacle
	std::vector<std::vector<double>> obs_list;
	std::vector<std::vector<std::vector<double>>> obs_list_points;

	// collision checker
	// from http://www.jeffreythompson.org/collision-detection/line-rect.php
	bool lineLine(double x1, double y1, double x2, double y2, double x3, double y3, double x4, double y4);
};


#endif