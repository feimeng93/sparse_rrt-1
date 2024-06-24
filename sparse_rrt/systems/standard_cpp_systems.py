from sparse_rrt import _sst_module
import numpy as np


class WithEuclideanDistanceComputer(object):
    """
    Add euclidian distance computer to a cpp system class
    """

    def distance_computer(self):
        return _sst_module.euclidean_distance(np.array(self.is_circular_topology()))


class Pendulum(_sst_module.Pendulum, WithEuclideanDistanceComputer):
    pass

class Point(_sst_module.Point, WithEuclideanDistanceComputer):
    pass


# class TwoLinkAcrobot(_sst_module.TwoLinkAcrobot):
#     """
#     Acrobot has its own custom distance for faster convergence
#     """

#     def distance_computer(self):
#         return _sst_module.TwoLinkAcrobotDistance()


# class CartPoleObs(_sst_module.RectangleObsSystem):
#     def __init__(self, obs_list, width):
#         super(CartPoleObs, self).__init__(obs_list, width, "cartpole_obs")
#     def distance_computer(self):
#         return _sst_module.euclidean_distance(np.array(self.is_circular_topology()))

# class TwoLinkAcrobotObs(_sst_module.RectangleObsSystem):
#     def __init__(self, obs_list, width):
#         super(TwoLinkAcrobotObs, self).__init__(obs_list, width, "acrobot_obs")
#     def distance_computer(self):
#         return _sst_module.TwoLinkAcrobotDistance()


class RectangleObs3D(_sst_module.RectangleObs3DSystem):
    def __init__(self, obstacle_list, obstacle_width, env_name):
        super().__init__(obstacle_list, obstacle_width, env_name)
        self.env_name = env_name

    def distance_computer(self):
        if self.env_name == "Nonlinear3D":
            return _sst_module.PNLDistance()


class RectangleObs2D(_sst_module.RectangleObs2DSystem):
    def __init__(self, obstacle_list, obstacle_width, env_name):
        super().__init__(obstacle_list, obstacle_width, env_name)
        self.env_name = env_name

    def distance_computer(self):
        if self.env_name == "CartPole":
            return _sst_module.CartpoleDistance()
        elif self.env_name == "Linear2D":
            return _sst_module.LDistance()
        elif self.env_name == "TwoLinkRobot":
            return _sst_module.TwoLinkAcrobotDistance()
        elif self.env_name == "DampingPendulum":
            return _sst_module.PendulumDistance()
        if self.env_name == "PlanarQuadrotor":
            return _sst_module.QuadrotorDistance()
        else:
            raise ValueError("Invalid env_name.")
