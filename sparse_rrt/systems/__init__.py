from . import standard_cpp_systems
from sparse_rrt.systems.pendulum import Pendulum

# List of standard systems that come out of the box
_standard_system_classes = {
    'CartPole': standard_cpp_systems.RectangleObs2D,
    'Linear2D': standard_cpp_systems.RectangleObs2D,
    'Nonlinear3D': standard_cpp_systems.RectangleObs3D,
    'DampingPendulum': standard_cpp_systems.Pendulum,
    'TwoLinkRobot': standard_cpp_systems.RectangleObs2D,
    'PlanarQuadrotor': standard_cpp_systems.RectangleObs2D,
    'py_pendulum': Pendulum,
    'py_acrobot': standard_cpp_systems.RectangleObs2D,
    'point': standard_cpp_systems.Point
}


def create_standard_system(system_name, *args, **kwargs):
    '''
    Create standard system by string identifier
    :param system_name: string, a name of the system from _standard_system_classes
    :param args: construction args of the system
    :param kwargs: construction kwargs of the system
    :return: A system that supports ISystem
    '''
    return _standard_system_classes[system_name](*args, **kwargs)
