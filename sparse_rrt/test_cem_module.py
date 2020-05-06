import sys
sys.path.append('/media/arclabdl1/HD1/Linjun/mpc-mpnet-py/deps/sparse_rrt-1')
from sparse_rrt.systems import standard_cpp_systems
import numpy as np


system = standard_cpp_systems.TwoLinkAcrobotObs(np.array([[12., 13.],[1., 5.]]), 6)
