/**
 * @file point.hpp
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

#ifndef SPARSE_PLANARQUADROTOR_HPP
#define SPARSE_PLANARQUADROTOR_HPP

#include "systems/system.hpp"


/**
 * @brief A simple system implementing a 2d point. 
 * @details A simple system implementing a 2d point. It's controls include velocity and direction.
 */
class planar_quadrotor_t : public system_t
{
public:
	planar_quadrotor_t(std::vector<std::vector<double>>& _obs_list, double width)
	{
		state_dimension = 6;
		control_dimension = 2;
		temp_state = new double[state_dimension]();
		deriv = new double[state_dimension]();
		u = new double[control_dimension]();
		validity = true;
		// copy the items from _obs_list to obs_list
		for(unsigned i=0;i<_obs_list.size();i++)
		for(unsigned int oi = 0; oi < _obs_list.size(); oi++){
			std::vector<double> min_max_i = {_obs_list.at(oi).at(0) - width / 2, _obs_list.at(oi).at(0) + width / 2,
											 _obs_list.at(oi).at(1) - width / 2, _obs_list.at(oi).at(1) + width / 2
											};// size = 4
			obs_min_max.push_back(min_max_i); // size = n_o (* 4)
		}

	}
	virtual ~planar_quadrotor_t(){ delete[] temp_state; delete[] deriv; delete[] u ;}

	static double distance(const double* point1, const double* point2, unsigned int);

	/**
	 * @copydoc system_t::propagate(double*, double*, int, int, double*, double& )
	 */
	virtual bool propagate(
	    const double* start_state, unsigned int state_dimension,
        const double* control, unsigned int control_dimension,
	    int num_steps, double* result_state, double integration_step) override;

	/**
	 * @copydoc system_t::enforce_bounds()
	 */
	virtual void enforce_bounds() override;
	
	/**
	 * @copydoc system_t::valid_state()
	 */
	virtual bool valid_state() override;

	/**
	 * @copydoc system_t::visualize_point(double*, svg::Dimensions)
	 */
	std::tuple<double, double> visualize_point(const double* state, unsigned int state_dimension) const override;

	/**
	 * @copydoc system_t::visualize_obstacles(svg::DocumentBody&, svg::Dimensions)
	 */
    std::string visualize_obstacles(int image_width, int image_height) const override;

	/**
	 * @copydoc system_t::get_state_bounds()
	 */
    std::vector<std::pair<double, double>> get_state_bounds() const override;

    /**
	 * @copydoc system_t::get_control_bounds()
	 */
    std::vector<std::pair<double, double>> get_control_bounds() const override;

	/**
	 * @copydoc enhanced_system_t::is_circular_topology()
	 */
    std::vector<bool> is_circular_topology() const override;

	// 	/**
	//  * normalize state to [-1,1]^13
	//  */
	// void normalize(const double* state, double* normalized);
	
	// /**
	//  * denormalize state back
	//  */
	// void denormalize(double* normalized,  double* state);
	
	// /**
	//  * get loss for cem-mpc solver
	//  */
	// double get_loss(double* point1, const double* point2, double* weight);

protected:
	double* deriv;
	void update_derivative(const double* control);
	double *u;
	bool validity;
	std::vector<std::vector<double>> obs_min_max;
};


#endif