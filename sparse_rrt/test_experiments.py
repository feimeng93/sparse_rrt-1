import sys
sys.path.append('/media/arclabdl1/HD1/Linjun/mpc-mpnet-py/deps/sparse_rrt-1')
sys.path.append('.')
from sparse_rrt.experiments import standard_experiments
from sparse_rrt.experiments.experiment_utils import run_config


def test_experiments():
    '''
    Test that all experiments configs are working
    '''
    for name, config in standard_experiments.iteritems():
        print("Running %s..." % name)
        config = config.copy()
        config['number_of_iterations'] = 100
        config['display_type'] = None
        config['debug_period'] = 50
        run_config(config)


if __name__ == '__main__':
    test_experiments()
