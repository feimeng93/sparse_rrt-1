from sparse_rrt import _sst_module

from sparse_rrt.visualization import render_svg, svg_header, svg_footer


def visualize_wrapper(parent_class):
    '''
    Class factory that wraps visualization function for planners
    :param parent_class: a planner class (RRTWrapper or SSTWrapper)
    :return: a class with visualization functions
    '''
    class VisualizeWrapper(parent_class):
        def visualize_tree(self, system, image_width=500, image_height=500):
            body_string = parent_class.visualize_tree(
                self,
                system,
                image_width=image_width,
                image_height=image_height
            )
            body_string += system.visualize_obstacles(image_width, image_height)
            return render_svg(svg_header(width=image_width, height=image_height) + body_string + svg_footer())

        def visualize_nodes(self, system, image_width=800, image_height=800, node_diameter=4, solution_node_diameter=8):
            body_string = parent_class.visualize_nodes(
                self,
                system,
                image_width=image_width,
                image_height=image_height,
                node_diameter = node_diameter,
                solution_node_diameter=solution_node_diameter
            )
            body_string += system.visualize_obstacles(image_width, image_height)
            return render_svg(svg_header(width=image_width, height=image_height) + body_string + svg_footer())

    return VisualizeWrapper


class SST(visualize_wrapper(_sst_module.SSTWrapper)):
    '''
    Sparse stable trees planner
    '''
    pass

class SST_Backend(visualize_wrapper(_sst_module.SSTBackendWrapper)):
    '''
    Sparse stable trees planner
    '''
    pass

# class RRT(visualize_wrapper(_sst_module.RRTWrapper)):
#     '''
#     RRT planner (baseline)
#     '''
#     pass
