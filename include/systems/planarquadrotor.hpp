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
        {
            // each obstacle is represented by its middle point
            std::vector<std::vector<double>> obs(4, std::vector<double>(2));
            // calculate the four points representing the rectangle in the order
            // UL, UR, LR, LL
            // the obstacle points are concatenated for efficient calculation
            double x = _obs_list[i][0];
            double y = _obs_list[i][1];
			// order: (left-bottom, right-bottom, right-upper, left-upper)

            obs[0][0] = x - width / 2;  obs[0][1] = y - width / 2;
            obs[1][0] = x + width / 2;  obs[1][1] = y - width / 2;
            obs[2][0] = x + width / 2;  obs[2][1] = y + width / 2;
            obs[3][0] = x - width / 2;  obs[3][1] = y + width / 2;
            obs_list.push_back(obs);

			// horizontal axis and vertical
            std::vector<std::vector<double>> obs_axis_i(2, std::vector<double> (2, 0));
            obs_axis_i[0][0] = obs[1][0] - obs[0][0]; // width
            obs_axis_i[1][0] = obs[3][0] - obs[0][0]; //0
            obs_axis_i[0][1] = obs[1][1] - obs[0][1]; //0
            obs_axis_i[1][1] = obs[3][1] - obs[0][1]; //width

            std::vector<double> obs_length;
            obs_length.push_back(sqrt(obs_axis_i[0][0]*obs_axis_i[0][0]+obs_axis_i[0][1]*obs_axis_i[0][1]));
            obs_length.push_back(sqrt(obs_axis_i[1][0]*obs_axis_i[1][0]+obs_axis_i[1][1]*obs_axis_i[1][1]));
			// ormalize the axis
            for (unsigned i1=0; i1<2; i1++)
            {
                for (unsigned j1=0; j1<2; j1++)
                {
                    obs_axis_i[i1][j1] = obs_axis_i[i1][j1] / obs_length[i1];
                }
            }
            obs_axis.push_back(obs_axis_i);

			// obtain the inner product of the left-bottom corner with the axis to obtain the minimal of projection value
            std::vector<double> obs_ori_i;
            obs_ori_i.push_back(obs[0][0]*obs_axis_i[0][0]+ obs[0][1]*obs_axis_i[0][1]);
            obs_ori_i.push_back(obs[0][0]*obs_axis_i[1][0]+ obs[0][1]*obs_axis_i[1][1]);

            obs_ori.push_back(obs_ori_i);
        }

	}
	virtual ~planar_quadrotor_t(){ delete[] temp_state; delete[] deriv; delete[] u ;}

	bool overlap(std::vector<std::vector<double>>& b1corner, std::vector<std::vector<double>>& b1axis,
	             std::vector<double>& b1orign, std::vector<double>& b1ds,
				 std::vector<std::vector<double>>& b2corner, std::vector<std::vector<double>>& b2axis,
				 std::vector<double>& b2orign, std::vector<double>& b2ds);

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
	std::vector<std::vector<std::vector<double>>> obs_list;
	double obs_width;
    std::vector<std::vector<std::vector<double>>> obs_axis;
    std::vector<std::vector<double>> obs_ori;
};


#endif