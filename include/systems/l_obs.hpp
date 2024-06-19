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

#ifndef SPARSE_L_OBS_HPP
#define SPARSE_L_OBS_HPP

#include "systems/enhanced_system.hpp"

// class Rectangle_t
// {
// public:
// 	/**
// 	 * @brief Create a rectangle using two corners
// 	 * 
// 	 * @param lx Bottom Left X coordinate
// 	 * @param ly Bottom Left Y coordinate
// 	 * @param hx Top Right X coordinate
// 	 * @param hy Top Right Y coordinate
// 	 */
// 	Rectangle_t(double lx,double ly,double hx,double hy)
// 	{
// 		low_x = lx;
// 		low_y = ly;
// 		high_x = hx;
// 		high_y = hy;
// 	}
// 	/**
// 	 * @brief Create a rectangle with center position and dimensions.
// 	 *
// 	 * @param pos_x Center X coordinate.
// 	 * @param pos_y Center Y coordinate.
// 	 * @param dim_x X Dimension
// 	 * @param dim_y Y Dimension
// 	 * @param value Just a flag to denote which constructor is used.
// 	 */
// 	Rectangle_t(double pos_x,double pos_y,double dim_x,double dim_y,bool value)
// 	{
// 		low_x = pos_x-dim_x/2;
// 		low_y = pos_y-dim_y/2;
// 		high_x = pos_x+dim_x/2;
// 		high_y = pos_y+dim_y/2;
// 	}
// 	double low_x;
// 	double low_y;
// 	double high_x;
// 	double high_y;
// };

/**
 * @brief A simple system implementing a 2d point. 
 * @details A simple system implementing a 2d point. It's controls include velocity and direction.
 */
class l_obs_t : public enhanced_system_t
{
public:
	l_obs_t(std::vector<std::vector<double>>& _obs_list, double width)
	{
		state_dimension = 2;
		control_dimension = 1;
		temp_state = new double[state_dimension];
		deriv = new double[state_dimension];
		// copy the items from _obs_list to obs_list
		for(unsigned i=0;i<_obs_list.size();i++)
		{
			// each obstacle is represented by its middle point
			std::vector<double> obs(4*2);
			// calculate the four points representing the rectangle in the order
			// UL, UR, LR, LL, clockwise
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
	virtual ~l_obs_t(){ delete[] temp_state; delete[] deriv; obs_list.clear();}

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

		/**
	 * normalize state to [-1,1]^13
	 */
	void normalize(const double* state, double* normalized);
	
	/**
	 * denormalize state back
	 */
	void denormalize(double* normalized,  double* state);
	
	/**
	 * get loss for cem-mpc solver
	 */
	double get_loss(double* point1, const double* point2, double* weight);

	

protected:
	double* deriv;
	void update_derivative(const double* control);
	std::vector<std::vector<double>> obs_list;
};


#endif