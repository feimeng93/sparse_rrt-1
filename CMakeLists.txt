cmake_minimum_required(VERSION 3.9)
project(SparseRRT LANGUAGES CXX CUDA)

# enable_language(CUDA)

# appending torch and cuda settings
set (CMAKE_CUDA_FLAGS "-std=c++11")  # this line is important to compile CUDA with CXX in gcc-5
list(APPEND CMAKE_PREFIX_PATH
    /media/arclabdl1/HD1/Linjun/external/libtorch
    /usr/local/cuda
    )
set(CUDA_TOOLKIT_ROOT_DIR "/usr/local/cuda")
find_package(Torch REQUIRED)
find_package(CUDA QUIET REQUIRED)



set(EXECUTABLE_OUTPUT_PATH ${PROJECT_SOURCE_DIR}/bin)
set(LIBRARY_OUTPUT_PATH ${PROJECT_SOURCE_DIR}/lib)
set(CMAKE_CXX_FLAGS "-std=c++1y ${CMAKE_CXX_FLAGS}")
set(CMAKE_POSITION_INDEPENDENT_CODE ON)

set(CMAKE_CXX_STANDARD 11)
 
set(INCLUDE_DIRECTORIES include)

add_definitions(-O3)
# SET(CMAKE_BUILD_TYPE "Debug")
#SET(CMAKE_CXX_FLAGS_DEBUG "$ENV{CXXFLAGS} -O0 -Wall -g2 -ggdb")
#SET(CMAKE_CXX_FLAGS_RELEASE "$ENV{CXXFLAGS} -O3 -Wall")


IF(APPLE)
   set(CMAKE_MACOSX_RPATH 1)
ENDIF(APPLE)

# Instead of find_package(PythonLibs REQUIRED), we manually search for python to make sure we link to the active version
# (as opposed to system python if brew python is installed)
execute_process(
    COMMAND "python" -c "import sys;print(sys.prefix)"
    OUTPUT_VARIABLE PYTHON_PREFIX
    OUTPUT_STRIP_TRAILING_WHITESPACE
)
#if(APPLE)
#    set(PYTHON_LIBRARIES "${PYTHON_PREFIX}/lib/libpython2.7.dylib")
#else(APPLE)
# # it seems like libpython2.7.so is not necessary on ubuntu
#   # set(PYTHON_LIBRARIES "${PYTHON_PREFIX}/lib/libpython2.7.so")
    #endif(APPLE)

execute_process(
    COMMAND "python" -c "from distutils import sysconfig; print(sysconfig.get_python_inc())"
    OUTPUT_VARIABLE PYTHON_INCLUDE_DIRS
    OUTPUT_STRIP_TRAILING_WHITESPACE
)


include_directories(
    # ${Numpy_INCLUDE_DIRS}  # this should come before PYTHON includes because it also contains system numpy
    ${PYTHON_INCLUDE_DIRS}
    "${PROJECT_SOURCE_DIR}/include"
    "${PROJECT_SOURCE_DIR}/deps/pybind11/include"
    # #torch
    ${TORCH_INCLUDE_DIRS}
    )

set(PLANNING_UTILS
    src/nearest_neighbors/graph_nearest_neighbors.cpp

    src/utilities/timer.cpp
    src/utilities/random.cpp
    src/image_creation/svg_image.cpp
    src/image_creation/planner_visualization.cpp

)

# for SST
set(SOURCE_FILES
    src/motion_planners/rrt.cpp
    src/motion_planners/sst.cpp
    src/motion_planners/sst_backend.cpp
    src/systems/car.cpp
    src/systems/cart_pole.cpp
    src/systems/pendulum.cpp
    src/systems/point.cpp
    src/systems/rally_car.cpp
    src/systems/two_link_acrobot.cpp
    src/systems/quadrotor.cpp
    src/systems/distance_functions.cpp
    ${PLANNING_UTILS}
)

add_library(${PROJECT_NAME} STATIC
    ${SOURCE_FILES})

add_library(sst_module SHARED
    ${PROJECT_SOURCE_DIR}/src/python_wrapper.cpp)
target_link_libraries(sst_module ${PYTHON_LIBRARIES} ${PROJECT_NAME})

# Don't prepend wrapper library name with lib and add to Python libs.
set_target_properties(sst_module PROPERTIES
        PREFIX "_"
        SUFFIX ".so"
        LIBRARY_OUTPUT_DIRECTORY ${PROJECT_SOURCE_DIR}/sparse_rrt
        )
set_property(TARGET sst_module PROPERTY CXX_STANDARD 11)

#set_source_files_properties(src/trajectory_optimizers/cem_cuda.cu PROPERTIES LANGUAGE CUDA)

# for mpc and cem
set(MPC_SOURCE
    src/systems/cart_pole_obs.cpp
    src/systems/two_link_acrobot_obs.cpp
    src/systems/quadrotor_obs.cpp
    src/systems/car_obs.cpp
    src/trajectory_optimizers/cem.cpp

    src/trajectory_optimizers/cem_cuda_cartpole.cu
    src/trajectory_optimizers/cem_cuda_acrobot.cu
    src/trajectory_optimizers/cem_cuda_car.cu
    src/trajectory_optimizers/cem_cuda_quadrotor.cu

    )

# set_property(TARGET CEMMPC PROPERTY CXX_STANDARD 11)

add_library(CEMMPC SHARED
    ${MPC_SOURCE}
    )

set(DEEP_SMP_MODULE
    ${PLANNING_UTILS}
    src/motion_planners/sst.cpp
    src/networks/mpnet.cpp
    src/networks/mpnet_cost.cpp
    src/motion_planners/deep_smp_mpc_sst.cpp
)


add_library(deep_smp_module SHARED
    ${PROJECT_SOURCE_DIR}/src/deep_smp_wrapper.cpp
    ${DEEP_SMP_MODULE}
    )
target_link_libraries(deep_smp_module ${PYTHON_LIBRARIES} ${TORCH_LIBRARIES} CEMMPC)

# Don't prepend wrapper library name with lib and add to Python libs.
set_target_properties(deep_smp_module PROPERTIES
        PREFIX "_"
        SUFFIX ".so"
        CXX_STANDARD 11

        LIBRARY_OUTPUT_DIRECTORY ${PROJECT_SOURCE_DIR}/sparse_rrt
        )
set_property(TARGET deep_smp_module PROPERTY CXX_STANDARD 11)


# for all test
### test trajectory optimizers MPC
add_executable(
    test_mpc
    tests/trajectory_optimizers/test_mpc.cpp
    )
target_link_libraries(test_mpc CEMMPC)


### test planners
add_executable(test_planner
    tests/motion_planners/test_planner.cpp
    ${DEEP_SMP_MODULE}
    )
target_link_libraries(
    test_planner   
    ${TORCH_LIBRARIES}
    CEMMPC
    )
set_property(TARGET test_planner PROPERTY CXX_STANDARD 14)

### test system
add_executable(test_system
    src/systems/cart_pole_obs.cpp
    src/systems/two_link_acrobot_obs.cpp
    src/systems/quadrotor_obs.cpp    tests/systems/test_system.cpp
    )



