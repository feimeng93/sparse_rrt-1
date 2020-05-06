#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include <pybind11/stl.h>
#include <pybind11/functional.h>

#include "systems/two_link_acrobot_obs_t.hpp"
#include "trajectory_optimizers/cem.hpp"
#include "motion_planners/deep_smp_mpc_sst.hpp"

namespace pybind11 {
    template <typename T>
    using safe_array = typename pybind11::array_t<T, pybind11::array::c_style>;
}

namespace py = pybind11;
using namespace pybind11::literals;

class __attribute__ ((visibility ("hidden"))) DSSTMPCWrapper{
    
public:

	/**
	 * @brief Python wrapper of DSSTMPC planner Constructor
	 * @details Python wrapper of DSSTMPC planner Constructor
	 *
	 */
    DSSTMPCWrapper(
            const py::safe_array<double> &start_state_array,
            const py::safe_array<double> &goal_state_array,
            double goal_radius,
            unsigned int random_seed,
            double sst_delta_near,
            double sst_delta_drain,
            const py::safe_array<double> &obs_list_array,
            double width,
            bool verbose
    ){
        auto start_state = start_state_array.unchecked<1>();
        auto goal_state = goal_state_array.unchecked<1>();

        if (obs_list_array.shape()[0] == 0) {
             throw std::runtime_error("Should contain at least one obstacles.");
         }
         if (obs_list_array.shape()[1] != 2) {
             throw std::runtime_error("Shape of the obstacle input should be (N,2).");
         }
         if (width <= 0.) {
             throw std::runtime_error("obstacle width should be non-negative.");
        }
        auto py_obs_list = obs_list_array.unchecked<2>();
        // initialize the array
        obs_list = std::vector<std::vector<double>>(obs_list_array.shape()[0], std::vector<double>(2, 0.0));
        for (unsigned int i = 0; i < obs_list.size(); i++) {
             obs_list[i][0] = py_obs_list(i, 0);
             obs_list[i][1] = py_obs_list(i, 1);
         }

        system = new two_link_acrobot_obs_t(obs_list, width);

        int ns = 32,
            nt = 5,
            ne = 4,
            max_it = 5;
        double converge_r = 1e-1,
            mu_u = 0,
            std_u = 4,
            mu_t = 0.1,
            std_t = 0.4,
            t_max = 0.5,
            step_size = 0.5;
        dt = 2e-2;
        
        cem.reset(
            new trajectory_optimizers::CEM(
                system, ns, nt,               
                ne, converge_r, 
                mu_u, std_u, 
                mu_t, std_t, t_max, 
                dt, loss_weights, max_it, verbose, step_size)
        );
       
        planner.reset(
            new deep_smp_mpc_sst_t(
                    &start_state(0), &goal_state(0), goal_radius,
                    system->get_state_bounds(),
                    system->get_control_bounds(),
                    two_link_acrobot_obs_t::distance,
                    random_seed,
                    sst_delta_near, sst_delta_drain, cem.get())
        );
    }

    // /**
	//  * @copydoc planner_t::step()
	//  */
    void step(int min_time_steps, int max_time_steps, double integration_step) {
        planner->step(system, min_time_steps, max_time_steps, integration_step);
    }

    void neural_step(py::safe_array<double>& obs_voxel_array) {
        auto obs_voxel_data = obs_voxel_array.unchecked<1>();
        std::vector<float> obs_vec;
        for (unsigned i=0; i < obs_voxel_data.shape(0); i++)
        {
            obs_vec.push_back(float(obs_voxel_data(i)));
        }
        //std::cout << "vector to torch obs vector.." << std::endl;
        torch::Tensor obs_tensor = torch::from_blob(obs_vec.data(), {1,1,32,32}).to(at::kCUDA);
        planner -> neural_step(system, dt, obs_tensor);

    }

    /**
    * @copydoc sst_backend_t::nearest_vertex()
    */
    py::object nearest_vertex(const py::safe_array<double> &sample_state_array){
        auto sample_state = sample_state_array.unchecked<1>();
        nearest = this -> planner -> nearest_vertex(&sample_state(0));
        const double* nearest_point = nearest -> get_point();
        
        py::safe_array<double> nearest_array({sample_state_array.shape()[0]});
        auto state_ref = nearest_array.mutable_unchecked<1>();
        for (unsigned int i = 0; i < sample_state_array.shape()[0]; i++){
            state_ref(i) = nearest_point[i];
        }
        return nearest_array;
    }

    void add_to_tree(const py::safe_array<double> &sample_state_array,
        double duration
    ){
        auto sample_state = sample_state_array.unchecked<1>();
        const double* sample_control = new double[planner -> get_control_dimension()];
        planner -> add_to_tree(&sample_state(0), sample_control, nearest, duration);       

    }
    /**
	 * @copydoc planner_t::get_solution()
	 */
    py::object get_solution() {
        std::vector<std::vector<double>> solution_path;
        std::vector<std::vector<double>> controls;
        std::vector<double> costs;
        planner->get_solution(solution_path, controls, costs);
        

        if (controls.size() == 0) {
            return py::none();
        }

        py::safe_array<double> controls_array({controls.size(), controls[0].size()});
        py::safe_array<double> costs_array({costs.size()});
        auto controls_ref = controls_array.mutable_unchecked<2>();
        auto costs_ref = costs_array.mutable_unchecked<1>();
        for (unsigned int i = 0; i < controls.size(); ++i) {
            for (unsigned int j = 0; j < controls[0].size(); ++j) {
                controls_ref(i, j) = controls[i][j];
            }
            costs_ref(i) = costs[i];
        }

        py::safe_array<double> state_array({solution_path.size(), solution_path[0].size()});
        auto state_ref = state_array.mutable_unchecked<2>();
        for (unsigned int i = 0; i < solution_path.size(); ++i) {
            for (unsigned int j = 0; j < solution_path[0].size(); ++j) {
                state_ref(i, j) = solution_path[i][j];
            }
        }
        return py::cast(std::tuple<py::safe_array<double>, py::safe_array<double>, py::safe_array<double>>
            (state_array, controls_array, costs_array));
    }

    /**
	 * @copydoc planner_t::get_number_of_nodes()
	 */
    unsigned int get_number_of_nodes() {
        return this->planner->get_number_of_nodes();
    }

protected:
    enhanced_system_t *system;
    std::unique_ptr<trajectory_optimizers::CEM> cem;
    std::unique_ptr<deep_smp_mpc_sst_t> planner;
    sst_node_t* nearest;
    std::vector<std::vector<double>> obs_list;
    double dt;
    double loss_weights[4] = {1, 1, 0.2, 0.2};


private:

	/**
	 * @brief Captured distance computer python object to prevent its premature death
	 */
    py::object  _distance_computer_py;
};

PYBIND11_MODULE(_deep_smp_module, m) {
    m.doc() = "Python wrapper for deep smp planners";
    py::class_<DSSTMPCWrapper>(m, "DSSTMPCWrapper")
        .def(py::init<const py::safe_array<double>&,
            const py::safe_array<double>&,
            double,
            unsigned int,
            double,
            double,
            const py::safe_array<double>&,
            double,
            bool>(),
                        
            "start_state"_a,
            "goal_state"_a,
            "goal_radius"_a,
            "random_seed"_a,
            "sst_delta_near"_a,
            "sst_delta_drain"_a,
            "obs_list"_a,
            "width"_a,
            "verbose"_a
        )
        .def("step", &DSSTMPCWrapper::step,
            "min_time_steps"_a,
            "max_time_steps"_a,
            "integration_step"_a
        )
        .def("nearest_vertex", &DSSTMPCWrapper::nearest_vertex,
            "sample_state_array"_a
        )
        .def("add_to_tree", &DSSTMPCWrapper::add_to_tree,
            "sample_state_array"_a,
            "duration"_a
        )
        .def("neural_step", &DSSTMPCWrapper::neural_step,
            "obs_voxel_array"_a
        )
        .def("get_solution", &DSSTMPCWrapper::get_solution)
        .def("get_number_of_nodes", &DSSTMPCWrapper::get_number_of_nodes)
        ;
}
