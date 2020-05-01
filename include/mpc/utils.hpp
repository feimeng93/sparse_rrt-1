#include <omplapp/apps/QuadrotorPlanning.h>
#include <ompl/base/StateValidityChecker.h>
#include <omplapp/config.h>
#include <ompl/control/planners/rrt/RRT.h>

using namespace ompl;

void get_svc(base::StateValidityCheckerPtr svc){
    app::QuadrotorPlanning setup;
    setup.setEnvironmentMesh("../assets/env1.dae");
    setup.setRobotMesh("../assets/quadrotor_scale.dae");
    base::StateSpacePtr stateSpace(setup.getStateSpace());

        // set the bounds for the R^3 part of SE(3)
    base::RealVectorBounds bounds(3);
    bounds.setLow(-10);
    bounds.setHigh(10);
    stateSpace->as<base::CompoundStateSpace>()->as<base::SE3StateSpace>(0)->setBounds(bounds);
    base::ScopedState<base::SE3StateSpace> start(setup.getGeometricComponentStateSpace());
    setup.setPlanner(std::make_shared<control::RRT>(setup.getSpaceInformation()));
    setup.setup();
    svc = setup.getStateValidityChecker();
}
