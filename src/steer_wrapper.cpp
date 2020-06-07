#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include <pybind11/stl.h>
#include <pybind11/functional.h>

#include "systems/enhanced_system.hpp"
#include "systems/two_link_acrobot_obs.hpp"
#include "systems/cart_pole_obs.hpp"

#include "trajectory_optimizers/cem.hpp"

namespace pybind11 {
    template <typename T>
    using safe_array = typename pybind11::array_t<T, pybind11::array::c_style>;
}

namespace py = pybind11;
using namespace pybind11::literals;

class py_system_interface : public enhanced_system_interface
{
public:

	/**
	 * @copydoc enhanced_system_interface::propagate()
	 */
    bool propagate(
        const double* start_state, unsigned int state_dimension,
        const double* control, unsigned int control_dimension,
        int num_steps,
        double* result_state, double integration_step) override
    {
        py::safe_array<double> start_state_array{{state_dimension}};
        std::copy(start_state, start_state + state_dimension, start_state_array.mutable_data(0));

        py::safe_array<double> control_array{{control_dimension}};
        std::copy(control, control + control_dimension, control_array.mutable_data(0));

        py::gil_scoped_acquire gil;
        py::function overload = py::get_overload(static_cast<const enhanced_system_interface *>(this), "propagate");
        if (!overload) {
            pybind11::pybind11_fail("Tried to call pure virtual function propagate");
            return false;
        }

        auto result = overload(start_state_array, control_array, num_steps, integration_step);
        if (py::isinstance<py::none>(result)) {
            return false;
        } else {
            auto result_state_array = py::detail::cast_safe<py::safe_array<double>>(std::move(result));
            std::copy(result_state_array.data(0), result_state_array.data(0) + state_dimension, result_state);
            return true;
        }
    }

	/**
	 * @copydoc enhanced_system_interface::visualize_point()
	 */
    std::tuple<double, double> visualize_point(const double* state, unsigned int state_dimension) const override {
        py::safe_array<double> state_array{{state_dimension}};
        std::copy(state, state + state_dimension, state_array.mutable_data(0));

        py::gil_scoped_acquire gil;
        py::function overload = py::get_overload(static_cast<const enhanced_system_interface *>(this), "visualize_point");
        if (!overload) {
            pybind11::pybind11_fail("Tried to call pure virtual function visualize_point");
            return std::make_tuple(-1., -1.);
        }
        auto result = overload(state_array);
        return py::detail::cast_safe<std::tuple<double, double>>(std::move(result));
    }

	/**
	 * @copydoc enhanced_system_interface::visualize_obstacles()
	 */
    std::string visualize_obstacles(int image_width, int image_height) const override
    {
    	PYBIND11_OVERLOAD(
            std::string,                /* Return type */
            enhanced_system_interface,           /* Parent class */
            visualize_obstacles,        /* Name of function in C++ (must match Python name) */
            image_width, image_height   /* Argument(s) */
        );
    }
};

class __attribute__ ((visibility ("hidden"))) RectangleObsWrapper : public enhanced_system_t
 {
 public:
    /**
     * @brief Python wrapper of RectangleObsWrapper constructor
     * @details Python wrapper of RectangleObsWrapper constructor
     *
     * @param _obs_list: numpy array (N x 2) representing the middle point of the obstacles
     * @param width: width of the rectangle obstacle
     */
    RectangleObsWrapper(
        const py::safe_array<double> &_obs_list,
        double width,
        const std::string &system_name){
            if (_obs_list.shape()[0] == 0) {
                throw std::runtime_error("Should contain at least one obstacles.");
            }
            if (_obs_list.shape()[1] != 2) {
                throw std::runtime_error("Shape of the obstacle input should be (N,2).");
            }
            if (width <= 0.) {
                throw std::runtime_error("obstacle width should be non-negative.");
            }
            auto py_obs_list = _obs_list.unchecked<2>();
            // initialize the array
            std::vector<std::vector<double>> obs_list(_obs_list.shape()[0], std::vector<double>(2, 0.0));
            // copy from python array to this array
            for (unsigned int i = 0; i < obs_list.size(); i++) {
                obs_list[i][0] = py_obs_list(i, 0);
                obs_list[i][1] = py_obs_list(i, 1);
            }
            if (system_name == "acrobot_obs"){
                system_obs.reset(new two_link_acrobot_obs_t(obs_list, width));
                state_dimension = 4;
                control_dimension = 1;
            } else if(system_name == "cartpole_obs"){
                system_obs.reset(new cart_pole_obs_t(obs_list, width));
                state_dimension = 4;
                control_dimension = 1;
            }
    }

    bool propagate(
        const double* start_state, unsigned int state_dimension,
        const double* control, unsigned int control_dimension,
        int num_steps, double* result_state, double integration_step){
            return system_obs->propagate(start_state, state_dimension, control, control_dimension,
                num_steps, result_state, integration_step);
    }
    // py::safe_array<double> propagate(
    //     const py::safe_array<double>& start_state,
    //     const py::safe_array<double>& control,
    //     int num_steps, double integration_step){
    //         auto start_state_array =  start_state.unchecked<1>();
    //         auto control_array =  control.unchecked<1>();
    //         double* result_state = new double[start_state.size()]();
    //         system_obs->propagate(start_state_array, start_state.size(), control_array, control.size(),
    //             num_steps, result_state, integration_step);
    //         py::safe_array<double> start_state_array{{start_state.size()}};
    //         std::copy(result_state, result_state + start_state.size(), start_state_array.mutable_data(0));
    //         return ;
    // }

    void enforce_bounds(){/*system_obs->enforce_bounds();*/}

    bool valid_state(){/*return system_obs->valid_state();*/}

    std::tuple<double, double> visualize_point(const double* state, unsigned int state_dimension) const override{
        return system_obs->visualize_point(state, state_dimension);
    }
    std::vector<std::pair<double, double>> get_state_bounds() const override{
        return system_obs->get_state_bounds();
    }
    std::vector<std::pair<double, double>> get_control_bounds() const override{
        return system_obs->get_control_bounds();
    }

    std::vector<bool> is_circular_topology() const override{
        return system_obs->is_circular_topology();
    }

    double get_loss(double* state, const double* goal, double* weight){
        return system_obs->get_loss(state, goal, weight);
    }

    void normalize(const double* state, double* normalized){
        system_obs->normalize(state, normalized);
    }

    void denormalize(double* normalized, double* state){
        system_obs->denormalize(normalized, state);
    }

 protected:
 	/**
 	 * @brief Created planner object
 	 */
     std::unique_ptr<enhanced_system_t> system_obs;
 };


// class __attribute__ ((visibility ("hidden"))) CEMMPCWrapper : public RectangleObsWrapper
// {
//  public:
//     /**
//      * @brief Python wrapper of RectangleObsWrapper constructor
//      * @details Python wrapper of RectangleObsWrapper constructor
//      *
//      * @param _obs_list: numpy array (N x 2) representing the middle point of the obstacles
//      * @param width: width of the rectangle obstacle
//      */
//     CEMMPCWrapper(
//         const py::safe_array<double> &_obs_list,
//         double width,
//         const std::string &system_name){
//             if (_obs_list.shape()[0] == 0) {
//                 throw std::runtime_error("Should contain at least one obstacles.");
//             }
//             if (_obs_list.shape()[1] != 2) {
//                 throw std::runtime_error("Shape of the obstacle input should be (N,2).");
//             }
//             if (width <= 0.) {
//                 throw std::runtime_error("obstacle width should be non-negative.");
//             }
//             auto py_obs_list = _obs_list.unchecked<2>();
//             // initialize the array
//             std::vector<std::vector<double>> obs_list(_obs_list.shape()[0], std::vector<double>(2, 0.0));
//             // copy from python array to this array
//             for (unsigned int i = 0; i < obs_list.size(); i++) {
//                 obs_list[i][0] = py_obs_list(i, 0);
//                 obs_list[i][1] = py_obs_list(i, 1);
//             }
//             if (system_name == "acrobot_obs"){
//                 system_obs.reset(new two_link_acrobot_obs_t(obs_list, width));
//                 state_dimension = 4;
//                 control_dimension = 1;
//             } else if(system_name == "cartpole_obs"){
//                 system_obs.reset(new cart_pole_obs_t(obs_list, width));
//                 state_dimension = 4;
//                 control_dimension = 1;
//             }
//     }

//     py::safe_array<double> solve(py::safe_array<double> strat, py::safe_array<double> goal){}
//  protected:
//  	/**
//  	 * @brief Created planner object
//  	 */
//      std::unique_ptr<enhanced_system_t> system_obs;
//  };


PYBIND11_MODULE(_steer_module, m) {
    m.doc() = "Python wrapper for deep smp planners";

    py::class_<enhanced_system_interface, py_system_interface> system_interface_var(m, "ISystem");
    system_interface_var
        .def(py::init<>())
        .def("propagate", &enhanced_system_interface::propagate)
        .def("visualize_point", &enhanced_system_interface::visualize_point)
        .def("visualize_obstacles", &enhanced_system_interface::visualize_obstacles);
    py::class_<enhanced_system_t> enhanced_system(m, "EnhancedSystem", system_interface_var);
    enhanced_system
        .def("get_state_bounds", &enhanced_system_t::get_state_bounds)
        .def("get_control_bounds", &enhanced_system_t::get_control_bounds)
        .def("is_circular_topology", &enhanced_system_t::is_circular_topology)
    ;

    py::class_<RectangleObsWrapper>(m, "RectangleObsSystem", enhanced_system)
        .def(py::init<const py::safe_array<double> &,
            double,
            const std::string &>(),
        "obstacle_list"_a,
        "obstacle_width"_a,
        "system_name"_a
    );

    // py::class_<two_link_acrobot_obs_t>(m, "TwoLinkAcrobotObs", enhanced_system).def(py::init<>());


    // py::class_<CEM>(m, "CEM")
    //     .def(py::init<const py::safe_array<double>&,
    //         const py::safe_array<double>&,
    //         double,
    //         unsigned int,
    //         double,
    //         double,
    //         const py::safe_array<double>&,
    //         double,
    //         bool>(),
                        
    //         "start_state"_a,
    //         "goal_state"_a,
    //         "goal_radius"_a,
    //         "random_seed"_a,
    //         "sst_delta_near"_a,
    //         "sst_delta_drain"_a,
    //         "obs_list"_a,
    //         "width"_a,
    //         "verbose"_a
    //     )
    //     ;
}
