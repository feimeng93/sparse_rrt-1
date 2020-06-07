/**
 * @file sst.cpp
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

#include "motion_planners/sst.hpp"
#include "motion_planners/deep_smp_mpc_sst.hpp"
#include "nearest_neighbors/graph_nearest_neighbors.hpp"
#ifndef TORCH_H
#include <torch/script.h>
#endif

#include <iostream>
#include <deque>

// #define DEBUG
// #define DEBUG_MPNET
// #define DEBUG_CEM
deep_smp_mpc_sst_t::deep_smp_mpc_sst_t(
    const double* in_start, const double* in_goal,
    double in_radius,
    const std::vector<std::pair<double, double> >& a_state_bounds,
    const std::vector<std::pair<double, double> >& a_control_bounds,
    std::function<double(const double*, const double*, unsigned int)> a_distance_function,
    unsigned int random_seed,
    double delta_near, double delta_drain,
    trajectory_optimizers::CEM* cem_ptr,
    networks::mpnet_cost_t *mpnet_ptr
    ) 
    : planner_t(in_start, in_goal, in_radius,
                a_state_bounds, a_control_bounds, a_distance_function, random_seed)
    , best_goal(nullptr)
    , sst_delta_near(delta_near)
    , sst_delta_drain(delta_drain)
    , cem_ptr(cem_ptr)
    , mpnet_ptr(mpnet_ptr)
{
    //initialize the metrics
    unsigned int state_dimensions = this->get_state_dimension();
    std::function<double(const double*, const double*)> raw_distance =
        [state_dimensions, a_distance_function](const double* s0, const double* s1) {
            return a_distance_function(s0, s1, state_dimensions);
        };
    metric.set_distance(raw_distance);

    root = new sst_node_t(in_start, a_state_bounds.size(), nullptr, tree_edge_t(nullptr, 0, -1.), 0.);
    metric.add_node(root);
    number_of_nodes++;

    samples.set_distance(raw_distance);

    sample_node_t* first_witness_sample = new sample_node_t(static_cast<sst_node_t*>(root), start_state, this->state_dimension);
    samples.add_node(first_witness_sample);
    witness_nodes.push_back(first_witness_sample);
}

deep_smp_mpc_sst_t::~deep_smp_mpc_sst_t() {
    delete root;
    for (auto w: this->witness_nodes) {
        delete w;
    }
}


void deep_smp_mpc_sst_t::get_solution(std::vector<std::vector<double>>& solution_path, std::vector<std::vector<double>>& controls, std::vector<double>& costs)
{
	if(best_goal==NULL)
		return;
	sst_node_t* nearest_path_node = best_goal;
	
	//now nearest_path_node should be the closest node to the goal state
	std::deque<sst_node_t*> path;
	while(nearest_path_node->get_parent()!=NULL)
	{
		path.push_front(nearest_path_node);
        nearest_path_node = nearest_path_node->get_parent();
	}

    std::vector<double> root_state;
    for (unsigned c=0; c<this->state_dimension; c++) {
        root_state.push_back(root->get_point()[c]);
    }
    solution_path.push_back(root_state);

	for(unsigned i=0;i<path.size();i++)
	{
        std::vector<double> current_state;
        for (unsigned c=0; c<this->state_dimension; c++) {
            current_state.push_back(path[i]->get_point()[c]);
        }
        solution_path.push_back(current_state);

        std::vector<double> current_control;
        for (unsigned c=0; c<this->control_dimension; c++) {
            current_control.push_back(path[i]->get_parent_edge().get_control()[c]);
        }
        controls.push_back(current_control);
        costs.push_back(path[i]->get_parent_edge().get_duration());
	}
}

void deep_smp_mpc_sst_t::step(enhanced_system_interface* system, int min_time_steps, int max_time_steps, double integration_step)
{
    /*
     * Generate a random sample
     * Find the closest existing node
     * Generate random control
     * Propagate for random time with constant random control from the closest node
     * If resulting state is valid, add a resulting state into the tree and perform sst-specific graph manipulations
     */
    double* sample_state = new double[this->state_dimension];
    double* sample_control = new double[this->control_dimension];
	this->random_state(sample_state);
	this->random_control(sample_control);
    sst_node_t* nearest = nearest_vertex(sample_state);
	int num_steps = this->random_generator.uniform_int_random(min_time_steps, max_time_steps);
    double duration = num_steps*integration_step;
	if(system->propagate(
	    nearest->get_point(), this->state_dimension, sample_control, this->control_dimension,
	    num_steps, sample_state, integration_step))
	{
		add_to_tree(sample_state, sample_control, nearest, duration);
	}
    delete sample_state;
    delete sample_control;
}


void deep_smp_mpc_sst_t::step(system_interface* system, int min_time_steps, int max_time_steps, double integration_step)
{
    /*
     * Generate a random sample
     * Find the closest existing node
     * Generate random control
     * Propagate for random time with constant random control from the closest node
     * If resulting state is valid, add a resulting state into the tree and perform sst-specific graph manipulations
     */
    double* sample_state = new double[this->state_dimension];
    double* sample_control = new double[this->control_dimension];
	this->random_state(sample_state);
	this->random_control(sample_control);
    sst_node_t* nearest = nearest_vertex(sample_state);
	int num_steps = this->random_generator.uniform_int_random(min_time_steps, max_time_steps);
    double duration = num_steps*integration_step;
	if(system->propagate(
	    nearest->get_point(), this->state_dimension, sample_control, this->control_dimension,
	    num_steps, sample_state, integration_step))
	{
		add_to_tree(sample_state, sample_control, nearest, duration);
	}
    delete sample_state;
    delete sample_control;
}


sst_node_t* deep_smp_mpc_sst_t::nearest_vertex(const double* sample_state)
{
	//performs the best near query
    std::vector<proximity_node_t*> close_nodes = metric.find_delta_close_and_closest(sample_state, this->sst_delta_near);

    double length = std::numeric_limits<double>::max();;
    sst_node_t* nearest = nullptr;
    for(unsigned i=0;i<close_nodes.size();i++)
    {
        tree_node_t* v = (tree_node_t*)(close_nodes[i]->get_state());
        double temp = v->get_cost() ;
        if( temp < length)
        {
            length = temp;
            nearest = (sst_node_t*)v;
        }
    }
    assert (nearest != nullptr);
    return nearest;
}

void deep_smp_mpc_sst_t::add_to_tree(const double* sample_state, const double* sample_control, sst_node_t* nearest, double duration)
{
	//check to see if a sample exists within the vicinity of the new node
    sample_node_t* witness_sample = find_witness(sample_state);

    sst_node_t* representative = witness_sample->get_representative();
	if(representative==NULL || representative->get_cost() > nearest->get_cost() + duration)
	{
		if(best_goal==NULL || nearest->get_cost() + duration <= best_goal->get_cost())
		{
			//create a new tree node
			//set parent's child
			sst_node_t* new_node = static_cast<sst_node_t*>(nearest->add_child(
			    new sst_node_t(
                    sample_state, this->state_dimension,
                    nearest,
                    tree_edge_t(sample_control, this->control_dimension, duration),
                    nearest->get_cost() + duration)
            ));
			number_of_nodes++;

            // std::cout <<"goal_distance:" << distance(new_node->get_point(), goal_state, this->state_dimension) << std::endl;
	        if(best_goal==NULL && this->distance(new_node->get_point(), goal_state, this->state_dimension)<goal_radius)
	        {
	        	best_goal = new_node;
	        	branch_and_bound((sst_node_t*)root);
	        }
	        else if(best_goal!=NULL && best_goal->get_cost() > new_node->get_cost() &&
	                this->distance(new_node->get_point(), goal_state, this->state_dimension)<goal_radius)
	        {
	        	best_goal = new_node;
	        	branch_and_bound((sst_node_t*)root);
	        }

            // Acquire representative again - it can be different
            representative = witness_sample->get_representative();
			if(representative!=NULL)
			{
				//optimization for sparsity
				if(representative->is_active())
				{
					metric.remove_node(representative);
					representative->make_inactive();
				}

	            sst_node_t* iter = representative;
	            while( is_leaf(iter) && !iter->is_active() && !is_best_goal(iter))
	            {
	                sst_node_t* next = (sst_node_t*)iter->get_parent();
	                remove_leaf(iter);
	                iter = next;
	            } 

			}
			witness_sample->set_representative(new_node);
			new_node->set_witness(witness_sample);
			metric.add_node(new_node);
		}
	}	

}

sample_node_t* deep_smp_mpc_sst_t::find_witness(const double* sample_state)
{
	double distance;
    sample_node_t* witness_sample = (sample_node_t*)samples.find_closest(sample_state, &distance)->get_state();
	if(distance > this->sst_delta_drain)
	{
		//create a new sample
		witness_sample = new sample_node_t(NULL, sample_state, this->state_dimension);
		samples.add_node(witness_sample);
		witness_nodes.push_back(witness_sample);
	}
    return witness_sample;
}

void deep_smp_mpc_sst_t::branch_and_bound(sst_node_t* node)
{
    // Copy children becuase apparently, they are going to be modified
    std::list<tree_node_t*> children = node->get_children();
    for (std::list<tree_node_t*>::const_iterator iter = children.begin(); iter != children.end(); ++iter)
    {
    	branch_and_bound((sst_node_t*)(*iter));
    }
    if(is_leaf(node) && node->get_cost() > best_goal->get_cost())
    {
    	if(node->is_active())
    	{
	    	node->get_witness()->set_representative(NULL);
	    	metric.remove_node(node);
	    }
    	remove_leaf(node);
    }
}

bool deep_smp_mpc_sst_t::is_leaf(tree_node_t* node)
{
	return node->is_leaf();
}

void deep_smp_mpc_sst_t::remove_leaf(sst_node_t* node)
{
	if(node->get_parent() != NULL)
	{
		node->get_parent_edge();
		node->get_parent()->remove_child(node);
		number_of_nodes--;
		delete node;
	}
}

bool deep_smp_mpc_sst_t::is_best_goal(tree_node_t* v)
{
	if(best_goal==NULL)
		return false;
    sst_node_t* new_v = best_goal;

    while(new_v->get_parent()!=NULL)
    {
        if(new_v == v)
            return true;

        new_v = new_v->get_parent();
    }
    return false;

}

void deep_smp_mpc_sst_t::neural_sample(enhanced_system_t* system, const double* nearest,
    double* neural_sample_state, torch::Tensor env_vox_tensor, bool refine, float refine_threshold, 
    bool using_one_step_cost, bool cost_reselection){
    mpnet_ptr->mpnet_sample(system, env_vox_tensor, nearest, goal_state, neural_sample_state, refine, refine_threshold, 
        using_one_step_cost, cost_reselection);
}

double deep_smp_mpc_sst_t::steer(enhanced_system_t* system, const double* start, const double* sample, 
    double* terminal_state, double integration_step){
    // TODO: add steer function with mpc
    #ifdef DEBUG_CEM 
    std::cout<<"start state:\t";
    for(unsigned int si = 0; si < this->state_dimension; si++){// save best state
            std::cout << start[si]<<","; 
    }
    std::cout<<"\ngoal state:\t";
    for(unsigned int si = 0; si < this->state_dimension; si++){// save best state
            std::cout << sample[si]<<","; 
    }
    std::cout << std::endl;
    #endif
    double* solution_u = new double[cem_ptr -> get_control_dimension()];
    double* solution_t = new double[cem_ptr -> get_num_step()];
    double* costs = new double[cem_ptr -> get_num_step()];
    double* state = new double[this->state_dimension];
    cem_ptr -> solve(start, sample, solution_u, solution_t);
    double duration = 0;
    for(unsigned int si = 0; si < this->state_dimension; si++){ //copy start state
        state[si] = start[si]; 
    }
    double min_loss = 1e3;// initialize logging variables
    unsigned int best_i = 0;

    for(unsigned int ti = 0; ti < cem_ptr -> get_num_step(); ti++){ // propagate
        if (system -> propagate(state, 
            this->state_dimension, 
            &solution_u[ti], 
            this->control_dimension, 
            (int)(solution_t[ti]/integration_step), 
            state,
            integration_step)){
               double current_loss = system -> get_loss(state, sample, cem_ptr -> weight);
                #ifdef DEBUG_CEM
                    std::cout<<"current_loss:" << current_loss <<std::endl;
                #endif
                costs[ti] = integration_step * (int)(solution_t[ti]/integration_step); // logging costs
                if (current_loss < min_loss || this->distance(state, goal_state, this->state_dimension) < goal_radius){//update min_loss
                    min_loss = current_loss;
                    best_i = ti;
                    for(unsigned int si = 0; si < this->state_dimension; si++){// save best state
                        terminal_state[si] = state[si]; 
                    }
                    if (min_loss < cem_ptr -> converge_radius){
                            break;
                    }
                    #ifdef DEBUG_CEM
                        std::cout<<"min_loss:" << min_loss <<std::endl;
                    #endif
                }        
        }
        else{
            break;
        } 
    }
    for(unsigned int ti = 0; ti <= best_i; ti++){    // compute duration until best duration
        duration += costs[ti];
    }

    #ifdef DEBUG_CEM
        std::cout<<"terminal state:\t";
        for(unsigned int si = 0; si < this->state_dimension; si++){// save best state
            std::cout << terminal_state[si]; 
        }
        std::cout<<std::endl;    
        std::cout << duration<< std::endl;
        std::cout<<"steered"<<std::endl;
    #endif
    delete solution_u;
    delete solution_t;
    delete state;
    delete costs;
    return duration;
}


void deep_smp_mpc_sst_t::neural_step(enhanced_system_t* system, double integration_step, torch::Tensor env_vox, 
    bool refine, float refine_threshold, bool using_one_step_cost, bool cost_reselection)
{
    //TODO: implement neural step
    /*
     * Generate a random sample
     * Find the closest existing node
     * apply neural sampling from this sample
     * connect the start node to the sample node with trajectory optimization
     * If resulting state is valid, add a resulting state into the tree and perform sst-specific graph manipulations
     */

    double* sample_state = new double[this->state_dimension]();
    double* neural_sample_state = new double[this->state_dimension]();
    double* terminal_state = new double[this->state_dimension]();
    double prob = this->random_generator.uniform_random(0, 1);
    if (prob < 0.05){
        for (unsigned int i = 0; i < state_dimension; i++){
            sample_state[i] = goal_state[i];
        }
    }
    else{
	    this->random_state(sample_state);
        sst_node_t* nearest = nearest_vertex(sample_state);
        //  add neural sampling 
        neural_sample(system, nearest->get_point(), neural_sample_state, env_vox, refine, refine_threshold, using_one_step_cost, cost_reselection); 
        // steer func
        
        double duration = steer(system, nearest->get_point(), neural_sample_state, terminal_state, integration_step);
        // std::cout<<"duration:" << duration << std::endl;    

        if(duration > 0)
        {
            add_to_tree(terminal_state, 0, nearest, duration);
        }
    }
    
    delete sample_state;
    delete neural_sample_state;
    delete terminal_state;
}


void deep_smp_mpc_sst_t::mpc_step(enhanced_system_t* system, double integration_step)
{
    /*
     * Generate a random sample
     * Find the closest existing node
     * Generate random control
     * Propagate for random time with constant random control from the closest node
     * If resulting state is valid, add a resulting state into the tree and perform sst-specific graph manipulations
     */
    double* terminal_state = new double[this->state_dimension]();
    double* sample_state = new double[this->state_dimension];
	this->random_state(sample_state);
    sst_node_t* nearest = nearest_vertex(sample_state);
    double duration = steer(system, nearest->get_point(), sample_state, terminal_state, integration_step);
	if(duration > 0)
	{
		add_to_tree(terminal_state, 0, nearest, duration);
	}
    delete sample_state;
    delete terminal_state;
}
