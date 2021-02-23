/**
 * @file sst.hpp
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

#ifndef SPARSE_PLANNER_SST_BACKEND_HPP
#define SPARSE_PLANNER_SST_BACKEND_HPP

#include "systems/system.hpp"
#include "motion_planners/planner.hpp"

#ifndef SPARSE_PLANNER_SST_HPP
#include "motion_planners/sst.hpp"
#endif

/**
 * @brief The motion planning algorithm SST (Stable Sparse-RRT)
 * @details The motion planning algorithm SST (Stable Sparse-RRT)
 */
class sst_backend_t : public planner_t
{
public:
	/**
	 * @brief SST planner Constructor
	 * @details SST planner Constructor
	 *
	 * @param in_start The start state.
	 * @param in_goal The goal state
	 * @param in_radius The radial size of the goal region centered at in_goal.
	 * @param a_state_bounds A vector with boundaries of the state space (min and max)
	 * @param a_control_bounds A vector with boundaries of the control space (min and max)
	 * @param distance_function Function that returns distance between two state space points
	 * @param random_seed The seed for the random generator
	 * @param delta_near Near distance threshold for SST
	 * @param delta_drain Drain distance threshold for SST
	 */
	sst_backend_t(const double* in_start, const double* in_goal,
	      double in_radius,
	      const std::vector<std::pair<double, double> >& a_state_bounds,
		  const std::vector<std::pair<double, double> >& a_control_bounds,
		  std::function<double(const double*, const double*, unsigned int)> distance_function,
		  unsigned int random_seed,
		  double delta_near, double delta_drain);
	virtual ~sst_backend_t();

	/**
	 * @copydoc planner_t::get_solution(std::vector<std::pair<double*,double> >&)
	 */
	virtual void get_solution(std::vector<std::vector<double>>& solution_path, std::vector<std::vector<double>>& controls, std::vector<double>& costs);
	
	/**
	 * @copydoc planner_t::step()
	 */
	 virtual void step(system_interface* system, int min_time_steps, int max_time_steps, double integration_step);
	
	// Expose two functions public to enable the python wrappers to call 
	/**
	 * @brief Finds a node to propagate from.
	 * @details Finds a node to propagate from. It does this through a procedure called BestNear w
	 * which examines a neighborhood around a randomly sampled point and returns the lowest cost one.
	 */
	sst_node_t* nearest_vertex(const double* sample_state);

	/**
	 * @brief If propagation was successful, add the new state to the tree.
	 * @details If propagation was successful, add the new state to the tree.
	 */
	void add_to_tree(const double* sample_state, const double* sample_control, sst_node_t* nearest, double duration);
protected:

    /**
     * @brief The nearest neighbor data structure.
     */
    graph_nearest_neighbors_t metric;

	/**
	 * @brief The best goal node found so far.
	 */
	sst_node_t* best_goal;



	/**
	 * @brief Check if the currently created state is close to a witness.
	 * @details Check if the currently created state is close to a witness.
	 */
	sample_node_t* find_witness(const double* sample_state);

	/**
	 * @brief Checks if this node has any children.
	 * @details Checks if this node has any children.
	 * 
	 * @param node The node to examine.
	 * @return True if no children, false if at least one child.
	 */
	bool is_leaf(tree_node_t* node);

	/**
	 * @brief Checks if this node is on the solution path.
	 * @details Checks if this node is on the solution path.
	 * 
	 * @param v The node to check
	 * @return True if on the solution path, false if not.
	 */
	bool is_best_goal(tree_node_t* v);

	/**
	 * @brief Removes a leaf node from the tree.
	 * @details Removes a leaf node from the tree.
	 * 
	 * @param node The node to remove.
	 */
	void remove_leaf(sst_node_t* node);

	/**
	 * @brief Branch out and prune planning tree
	 * @details Branch out and prune planning tree
	 *
	 * @param node The node from which to branch
	 */
	void branch_and_bound(sst_node_t* node);

	/**
	 * The nearest neighbor structure for witness samples.
	 */
	graph_nearest_neighbors_t samples;

	/**
	 * @brief Near distance threshold for SST
	 */
	double sst_delta_near;

	/**
	 * @brief Drain distance threshold for SST
	 */
	double sst_delta_drain;

	/**
	 * @brief Container for witness nodes (to avoid memory leaks)
	 */
    std::vector<sample_node_t*> witness_nodes;
};

#endif