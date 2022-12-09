import logging
from warnings import warn

# import altair as alt
import matplotlib.patches as patches
import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
from matplotlib.cm import get_cmap
from matplotlib.lines import Line2D
from matplotlib.path import Path
from more_itertools import unique_everseen
from matplotlib.colors import to_hex

from .geometry import (
    circos_radius,
    get_cartesian,
    group_theta,
    node_theta,
    text_alignment,
)
from .polcart import to_degrees
from .utils import (
    cmaps,
    infer_data_type,
    is_data_diverging,
    items_in_groups,
    n_group_colorpallet,
    num_discrete_groups,
    to_pandas_edges,
    to_pandas_nodes,
)


def despine(ax):
    for spine in ax.spines:
        ax.spines[spine].set_visible(False)
    plt.setp(ax.get_xticklabels(), visible=False)
    plt.setp(ax.get_yticklabels(), visible=False)
    ax.xaxis.set_visible(False)
    ax.yaxis.set_visible(False)
class BasePlot(object):
    """
    BasePlot: An extensible class for designing new network visualizations.

    The BasePlot constructor takes in a NetworkX graph object, and a series of
    keyword arguments specifying how nodes and edges should be styled and
    ordered.

    An optional `data_types` dictionary can be passed in to bypass data type
    inference.

    :param graph: A NetworkX graph object.
    :type graph: `nx.Graph`, `nx.DiGraph`, `nx.MultiGraph`, `nx.MultiDiGraph`

    :param node_order: The node attribute on which to specify the coloring of nodes.
    :type node_order: `dict_key` (often `str`)

    :param node_size: The node attribute on which to specify the size of nodes.
    :type node_size: `dict_key` (often `str`)

    :param node_grouping: The node attribute on which to specify the grouping position of nodes.
    :type node_grouping: `dict_key` (often `str`)

    :param group_order: The order in which the groups should be plotted.
    :type group_order: `dict_key` ('alphabetically', 'default')

    :param group_label_position: The position of the group label.
    :type group_label_position: `dict_key` ('beginning', 'middle', 'end')

    :param group_label_position: Boolean, whether the group labels should be the same color as the nodes.
    :type group_label_position: `bool`

    :param node_color: The node attribute on which to specify the colour of nodes.
    :type node_color: `dict_key` (often `str`)

    :param node_labels: Boolean, whether to use node objects as labels or not.
    :type node_labels: `bool`

    :param edge_width: The edge attribute on which to specify the width of edges.
    :type edge_with: `dict_key` (often `str`)

    :param edge_color: The edge attribute on which to specify the colour of edges.
    :type edge_color: `dict_key` (often `str`)

    :param data_types: A mapping of node and edge data types that are stored.
    :type data_types: `dict`

    :param nodeprops: A `matplotlib`-compatible `props` dictionary.
    :type nodeprops: `dict`

    :param edgeprops: A `matplotlib-compatible `props` dictioanry.
    :type edgeprops: `dict`

    :param fontsize: A text property for editing size of labels in graph
    :type fontsize: int

    :param fontfamily: A text property to define font family. Valid inputs ('serif', 'sans-serif', 'fantasy', 'monospace').
    :type fontfamily: string
    """  # noqa

    def __init__(
        self,
        graph,
        node_order=None,
        node_size=None,
        node_grouping=None,
        group_order="alphabetically",
        node_color=None,
        node_labels=None,
        edge_width=None,
        edge_color=None,
        data_types=None,
        nodeprops=None,
        edgeprops=None,
        node_label_color=False,
        group_label_position=None,
        group_label_color=False,
        fontsize=10,
        fontfamily="serif",
        legend_handles=None,
        **kwargs,
    ):
        super(BasePlot, self).__init__()
        # Set graph object
        self.graph = graph
        self.nodes = list(graph.nodes())  # keep track of nodes separately.
        self.edges = list(graph.edges())
        # Set node arrangement
        self.node_order = node_order
        self.node_grouping = node_grouping
        self.group_order = group_order
        self.group_and_sort_nodes()

        # Set node radius
        self.node_size = node_size
        if self.node_size:
            self.node_sizes = []
            self.compute_node_sizes()
        else:
            self.node_sizes = [1] * len(self.nodes)

        # Set node colors
        self.node_color = node_color
        self.sm = None  # sm -> for scalarmappable. See https://stackoverflow.com/questions/8342549/matplotlib-add-colorbar-to-a-sequence-of-line-plots  # noqa
        logging.debug("INIT: {0}".format(self.sm))
        if self.node_color:
            self.node_colors = []
            self.compute_node_colors()
        else:
            self.node_colors = ["blue"] * len(self.nodes)
        self.node_labels = node_labels

        # Set edge properties
        self.edge_width = edge_width
        if self.edge_width:
            self.edge_widths = []
            self.compute_edge_widths()
        else:
            self.edge_widths = [1] * len(self.edges)

        self.edge_color = edge_color
        if self.edge_color:
            self.edge_colors = []
            self.compute_edge_colors()
        else:
            self.edge_colors = ["black"] * len(self.edges)

        # Set data_types dictionary
        if not data_types:
            self.data_types = dict()
        else:
            self.check_data_types(data_types)
            self.data_types = data_types

        figsize = (6, 6)
        if "figsize" in kwargs.keys():
            figsize = kwargs["figsize"]
        self.figure = plt.figure(figsize=figsize)
        self.ax = self.figure.add_subplot(1, 1, 1)
        despine(self.ax)

        # We provide the following attributes that can be set by the end-user.
        # nodeprops are matplotlib patches properties.
        if nodeprops:
            self.nodeprops = nodeprops
        else:
            self.nodeprops = {"radius": 1}
        # edgeprops are matplotlib line properties. These can be set after
        # instantiation but before calling the draw() function.
        if edgeprops:
            self.edgeprops = edgeprops
        else:
            self.edgeprops = {"facecolor": "none", "alpha": 0.2}
        if node_label_color:
            self.node_label_color = self.node_colors
        else:
            self.node_label_color = ["black"] * len(self.nodes)

        # Compute each node's positions.
        self.compute_node_positions()

        # Conditionally compute node label positions.
        self.compute_node_label_positions()

        # set group properties
        self.group_label_position = group_label_position
        self.groups = []
        if group_label_position:
            self.compute_group_label_positions()
            if group_label_color:
                self.compute_group_colors()
            else:
                self.group_label_color = ["black"] * len(self.nodes)

        # set text properties
        valid_fonts = ["serif", "sans-serif", "fantasy", "monospace"]
        if fontfamily not in valid_fonts:
            raise ValueError(f"fontfamily should be one of {valid_fonts}")
        self.fontfamily = fontfamily
        self.fontsize = fontsize

    def check_data_types(self, data_types):
        """
        Checks the data_types passed into the Plot constructor and makes sure
        that the values passed in belong to 'ordinal', 'categorical', or
        'continuous'.

        :param data_types: A dictionary mapping of data types.
        :type data_types: `dict`
        """
        for k, v in data_types.items():
            assert v in ["ordinal", "categorical", "continuous"]

    def draw(self):
        """
        Draws the Plot to screen.

        If there is a continuous datatype for the nodes, it will be reflected
        in self.sm being constructed (in `compute_node_colors`). It will then
        automatically add in a colorbar to the plot and scale the plot axes
        accordingly.
        """
        self.draw_nodes()
        self.draw_edges()

        # note that self.groups only exists on condition
        # that group_label_position was given!
        if hasattr(self, "groups") and self.groups:
            self.draw_group_labels()
        logging.debug("DRAW: {0}".format(self.sm))
        if self.sm:
            self.figure.subplots_adjust(right=0.8)
            cax = self.figure.add_axes([0.85, 0.2, 0.05, 0.6])
            self.figure.colorbar(self.sm, cax=cax)
        self.ax.relim()
        self.ax.autoscale_view()
        self.ax.set_aspect("equal")

    def compute_node_colors(self):
        """Compute the node colors. Also computes the colorbar."""
        data = [self.graph.nodes[n][self.node_color] for n in self.nodes]

        if self.group_order == "alphabetically":
            data_reduced = sorted(list(set(data)))
        elif self.group_order == "default":
            data_reduced = list(unique_everseen(data))

        dtype = infer_data_type(data)
        n_grps = num_discrete_groups(data)

        if dtype == "categorical" or dtype == "ordinal":
            if n_grps <= 8:
                cmap = get_cmap(
                    cmaps["Accent_{0}".format(n_grps)].mpl_colormap
                )
            else:
                cmap = n_group_colorpallet(n_grps)
        elif dtype == "continuous" and not is_data_diverging(data):
            cmap = get_cmap(cmaps["continuous"].mpl_colormap)
        elif dtype == "continuous" and is_data_diverging(data):
            cmap = get_cmap(cmaps["diverging"].mpl_colormap)

        for d in data:
            idx = data_reduced.index(d) / n_grps
            self.node_colors.append(cmap(idx))

        # Add colorbar if required.ListedColormap
        logging.debug("length of data_reduced: {0}".format(len(data_reduced)))
        logging.debug("dtype: {0}".format(dtype))
        if len(data_reduced) > 1 and dtype == "continuous":
            self.sm = plt.cm.ScalarMappable(
                cmap=cmap,
                norm=plt.Normalize(
                    vmin=min(data_reduced),
                    vmax=max(data_reduced),  # noqa  # noqa
                ),
            )
            self.sm._A = []


    def compute_group_colors(self):
        """Computes the group colors according to node colors"""
        seen = set()
        self.group_label_color = [
            x for x in self.node_colors if not (x in seen or seen.add(x))
        ]

    def compute_edge_colors(self):
        """Compute the edge colors."""
        data = [self.graph.edges[n][self.edge_color] for n in self.edges]
        data_reduced = sorted(list(set(data)))

        dtype = infer_data_type(data)
        n_grps = num_discrete_groups(data)
        if dtype == "categorical" or dtype == "ordinal":
            if n_grps <= 8:
                cmap = get_cmap(
                    cmaps["Accent_{0}".format(n_grps)].mpl_colormap
                )
            else:
                cmap = n_group_colorpallet(n_grps)
        elif dtype == "continuous" and not is_data_diverging(data):
            cmap = get_cmap(cmaps["weights"])

        for d in data:
            idx = data_reduced.index(d) / n_grps
            self.edge_colors.append(cmap(idx))
        # Add colorbar if required.
        logging.debug("length of data_reduced: {0}".format(len(data_reduced)))
        logging.debug("dtype: {0}".format(dtype))
        if len(data_reduced) > 1 and dtype == "continuous":
            self.sm = plt.cm.ScalarMappable(
                cmap=cmap,
                norm=plt.Normalize(
                    vmin=min(data_reduced),
                    vmax=max(data_reduced),  # noqa  # noqa
                ),
            )
            self.sm._A = []

    def compute_node_sizes(self):
        """Compute the node sizes."""
        if isinstance(self.node_size, str):
            nodes = self.graph.nodes
            self.node_sizes = [nodes[n][self.node_size] for n in self.nodes]
        else:
            self.node_sizes = self.node_size


    def compute_edge_widths(self):
        """Compute the edge widths."""
        if isinstance(self.edge_width, str):
            edges = self.graph.edges
            self.edge_widths = [edges[n][self.edge_width] for n in self.edges]
        else:
            self.edge_widths = self.edge_width

    def compute_group_label_positions(self):
        """Computes the position of each group label according to the wanted
        position ("beginning", "middle", "end")

        Needs to be implemented for each plot type
        """
        pass

    def compute_node_positions(self):
        """
        Computes the positions of each node on the plot.

        Needs to be implemented for each plot type.
        """
        pass

    def compute_node_label_positions(self):
        """
        Computes the positions of each node's labels on the plot. The
        horizontal and vertical alignment of the text varies according to the
        location.

        Needs to be implemented for each plot type.
        """
        pass

    def draw_nodes(self):
        """
        Renders the nodes to the plot or screen.

        Needs to be implemented for each plot type.
        """
        pass

    def draw_edges(self):
        """
        Renders the nodes to the plot or screen.

        Needs to be implemented for each plot type.
        """
        pass


    def draw_group_labels(self):
        """
        Renders the group labels to the plot or screen.

        Needs to be implemented for each plot type.
        """
        pass


    def group_and_sort_nodes(self):
        """
        Groups and then sorts the nodes according to the criteria passed into
        the Plot constructor.
        """
        if self.node_grouping and not self.node_order:
            if self.group_order == "alphabetically":
                self.nodes = [
                    n
                    for n, d in sorted(
                        self.graph.nodes(data=True),
                        key=lambda x: x[1][self.node_grouping],
                    )
                ]

            elif self.group_order == "default":
                grp = [
                    d[self.node_grouping]
                    for _, d in self.graph.nodes(data=True)
                ]
                grp_name = list(unique_everseen(grp))
                nodes = []
                for key in grp_name:
                    nodes.extend(
                        [
                            n
                            for n, d in self.graph.nodes(data=True)
                            if key in d.values()
                        ]
                    )
                self.nodes = nodes

        elif self.node_order and not self.node_grouping:
            self.nodes = [
                n
                for n, _ in sorted(
                    self.graph.nodes(data=True),
                    key=lambda x: x[1][self.node_order],
                )
            ]

        elif self.node_grouping and self.node_order:
            if self.group_order == "alphabetically":
                self.nodes = [
                    n
                    for n, d in sorted(
                        self.graph.nodes(data=True),
                        key=lambda x: (
                            x[1][self.node_grouping],
                            x[1][self.node_order],
                        ),
                    )
                ]
            elif self.group_order == "default":
                grp = [
                    d[self.node_grouping]
                    for _, d in self.graph.nodes(data=True)
                ]
                grp_name = list(unique_everseen(grp))
                nodes = []
                for key in grp_name:
                    nodes.extend(
                        [
                            n
                            for n, d in sorted(
                                self.graph.nodes(data=True),
                                key=lambda x: x[1][self.node_order],
                            )
                            if key in d.values()
                        ]
                    )
                self.nodes = nodes

class CircosPlot(BasePlot):
    """
    Plotting object for CircosPlot.

    Accepts the following additional arguments apart from the ones in
    `BasePlot`:

    :param node_label_layout: which/whether (a) node layout is used,
        either 'rotation', 'numbers' or None
    :type node_label_layout: `string`

    :param group_label_offset: how much to offset the group labels, so that
        they are not overlapping with node labels.
    :type group_label_offset: `float` or `int`

    :param group_legend: Whether to include the legend of the group labels.
    :type group_legend: `bool`
    """

    def __init__(self, graph, **kwargs):
        """Create the CircosPlot."""

        # A CircosPlot only makes sense for at least 3 nodes
        assert len(graph.nodes) >= 3
        # Node labels are specified in the node_label_layout argument
        specified_layout = kwargs.pop("node_label_layout", None)
        # Verify that the provided input is legitimate
        valid_node_label_layouts = (None, "rotation", "numbers")
        assert specified_layout in valid_node_label_layouts
        # Store the node label layout
        self.node_label_layout = specified_layout

        # Group labels' radius can be offset by a certain amount
        group_label_offset = kwargs.pop("group_label_offset", 0)
        assert group_label_offset >= 0
        # Store the group label offset
        self.group_label_offset = group_label_offset

        #
        super(CircosPlot, self).__init__(graph, **kwargs)

        # Add legend to plot
        if "group_legend" in kwargs.keys():
            if kwargs["group_legend"]:
                self.draw_legend()

    def compute_group_label_positions(self):
        """
        Computes the x,y positions of the group labels.
        """
        assert self.group_label_position in ["beginning", "middle", "end"]
        data = [self.graph.nodes[n][self.node_grouping] for n in self.nodes]
        node_length = len(data)
        groups = items_in_groups(data)

        edge_of_plot = self.plot_radius + self.nodeprops["radius"]
        # The 1.02 serves as padding
        radius = 1.02 * edge_of_plot + self.group_label_offset
        xs = []
        ys = []
        has = []
        vas = []
        node_idcs = np.cumsum(list(groups.values()))
        node_idcs = np.insert(node_idcs, 0, 0)
        if self.group_label_position == "beginning":
            for idx in node_idcs[:-1]:
                x, y = get_cartesian(
                    r=radius, theta=group_theta(node_length, idx)
                )
                ha, va = text_alignment(x, y)
                xs.append(x)
                ys.append(y)
                has.append(ha)
                vas.append(va)

        elif self.group_label_position == "middle":
            node_idcs = node_idcs.reshape(len(node_idcs), 1)
            node_idcs = np.concatenate((node_idcs[:-1], node_idcs[1:]), axis=1)
            for idx in node_idcs:
                theta1 = group_theta(node_length, idx[0])
                theta2 = group_theta(node_length, idx[1] - 1)
                x, y = get_cartesian(r=radius, theta=(theta1 + theta2) / 2)
                ha, va = text_alignment(x, y)
                xs.append(x)
                ys.append(y)
                has.append(ha)
                vas.append(va)

        elif self.group_label_position == "end":
            for idx in node_idcs[1::]:
                x, y = get_cartesian(
                    r=radius, theta=group_theta(node_length, idx - 1)
                )
                ha, va = text_alignment(x, y)
                xs.append(x)
                ys.append(y)
                has.append(ha)
                vas.append(va)

        self.group_label_coords = {"x": xs, "y": ys}
        self.group_label_aligns = {"has": has, "vas": vas}
        self.groups = groups.keys()


    def compute_node_positions(self):
        """
        Uses the get_cartesian function to compute the positions of each node
        in the Circos plot.
        """
        xs = []
        ys = []
        node_r = self.nodeprops["radius"]
        radius = circos_radius(n_nodes=len(self.graph.nodes()), node_r=node_r)
        self.plot_radius = radius
        self.nodeprops["linewidth"] = radius * 0.01
        for node in self.nodes:
            x, y = get_cartesian(r=radius, theta=node_theta(self.nodes, node))
            xs.append(x)
            ys.append(y)
        self.node_coords = {"x": xs, "y": ys}


    def compute_node_label_positions(self):
        """
        Uses the get_cartesian function to compute the positions of each node
        label in the Circos plot.

        This method is always called after the compute_node_positions
        method, so that the plot_radius is pre-computed.
        This will also add a new attribute, `node_label_rotation` to the object
        which contains the rotation angles for each of the nodes. Together with
        the node coordinates this can be used to add additional annotations
        with rotated text.
        """
        self.init_node_label_meta()

        for node in self.nodes:

            # Define radius 'radius' and circumference 'theta'
            theta = node_theta(self.nodes, node)
            # multiplication factor 1.02 moved below
            radius = self.plot_radius + self.nodeprops["radius"]

            # Coordinates of text inside nodes
            if self.node_label_layout == "numbers":
                radius_adjustment = self.plot_radius / radius
            else:
                radius_adjustment = 1.02
            x, y = get_cartesian(r=radius * radius_adjustment, theta=theta)

            # ----- For numbered nodes -----

            # Node label x-axis coordinate
            tx, _ = get_cartesian(r=radius, theta=theta)
            # Create the quasi-circular positioning on the x axis
            tx *= 1 - np.log(np.cos(theta) * self.nonzero_sign(np.cos(theta)))
            # Move each node a little further away from the circos
            tx += self.nonzero_sign(x)

            # Node label y-axis coordinate numerator
            numerator = radius * (
                theta % (self.nonzero_sign(y) * self.nonzero_sign(x) * np.pi)
            )
            # Node label y-axis coordinate denominator
            denominator = self.nonzero_sign(x) * np.pi
            # Node label y-axis coordinate
            ty = 2 * (numerator / denominator)

            # ----- For rotated nodes -----

            # Computes the text rotation
            theta_deg = to_degrees(theta)
            if theta_deg >= -90 and theta_deg <= 90:  # right side
                rot = theta_deg
            else:  # left side
                rot = theta_deg - 180

            # Store values
            self.store_node_label_meta(x, y, tx, ty, rot)


    @staticmethod
    def nonzero_sign(xy):
        """
        A sign function that won't return 0
        """
        return -1 if xy < 0 else 1


    def init_node_label_meta(self):
        """
        This function ensures that self.node_label_coords
        exist with the correct keys and empty entries
        This function should not be called by the user
        """

        # Reset node label coorc/align dictionaries
        self.node_label_coords = {"x": [], "y": [], "tx": [], "ty": []}
        self.node_label_aligns = {"has": [], "vas": []}
        self.node_label_rotation = []


    def store_node_label_meta(self, x, y, tx, ty, rot):
        """
        This function stored coordinates-related metadate for a node
        This function should not be called by the user

        :param x: x location of node label or number
        :type x: np.float64

        :param y: y location of node label or number
        :type y: np.float64

        :param tx: text location x of node label (numbers)
        :type tx: np.float64

        :param ty: text location y of node label (numbers)
        :type ty: np.float64

        :param rot: rotation angle of the text (rotation)
        :type rot: float
        """

        # Store computed values
        self.node_label_coords["x"].append(x)
        self.node_label_coords["y"].append(y)
        self.node_label_coords["tx"].append(tx)
        self.node_label_coords["ty"].append(ty)

        # Computes the text alignment for x
        if x == 0:
            self.node_label_aligns["has"].append("center")
        elif x > 0:
            self.node_label_aligns["has"].append("left")
        else:
            self.node_label_aligns["has"].append("right")

        # Computes the text alignment for y
        if self.node_label_layout == "rotation" or y == 0:
            self.node_label_aligns["vas"].append("center")
        elif y > 0:
            self.node_label_aligns["vas"].append("bottom")
        else:
            self.node_label_aligns["vas"].append("top")

        self.node_label_rotation.append(rot)


    def draw_nodes(self):
        """
        Renders nodes to the figure.
        """
        node_r = self.nodeprops["radius"]
        lw = self.nodeprops["linewidth"]
        for i, node in enumerate(self.nodes):
            x = self.node_coords["x"][i]
            y = self.node_coords["y"][i]
            color = self.node_colors[i]
            node_patch = patches.Circle(
                (x, y), node_r, lw=lw, color=color, zorder=2
            )
            self.ax.add_patch(node_patch)
            if self.node_labels:
                label_x = self.node_label_coords["x"][i]
                label_y = self.node_label_coords["y"][i]
                label_tx = self.node_label_coords["tx"][i]
                label_ty = self.node_label_coords["ty"][i]
                label_ha = self.node_label_aligns["has"][i]
                label_va = self.node_label_aligns["vas"][i]

                # ----- Node label rotation layout -----

                if self.node_label_layout == "rotation":
                    rot = self.node_label_rotation[i]

                    self.ax.text(
                        s=node,
                        x=label_x,
                        y=label_y,
                        ha=label_ha,
                        va=label_va,
                        rotation=rot,
                        rotation_mode="anchor",
                        color=self.node_label_color[i],
                        fontsize=self.fontsize,
                        family=self.fontfamily,
                    )

                # ----- Node label numbering layout -----

                elif self.node_label_layout == "numbers":

                    # Draw descriptions for labels
                    desc = "%s - %s" % ((i, node) if (x > 0) else (node, i))
                    self.ax.text(
                        s=desc,
                        x=label_tx,
                        y=label_ty,
                        ha=label_ha,
                        va=label_va,
                        color=self.node_label_color[i],
                        fontsize=self.fontsize,
                        family=self.fontfamily,
                    )

                    # Add numbers to nodes
                    self.ax.text(
                        s=i, x=label_x, y=label_y, ha="center", va="center"
                    )

                # Standard node label layout
                else:

                    # Draw node text straight from the nodes
                    self.ax.text(
                        s=node,
                        x=label_x,
                        y=label_y,
                        ha=label_ha,
                        va=label_va,
                        color=self.node_label_color[i],
                        fontsize=self.fontsize,
                        family=self.fontfamily,
                    )


    def draw_edges(self):
        """
        Renders edges to the figure.
        """
        for i, (start, end) in enumerate(self.graph.edges()):
            start_theta = node_theta(self.nodes, start)
            end_theta = node_theta(self.nodes, end)
            verts = [
                get_cartesian(self.plot_radius, start_theta),
                (0, 0),
                get_cartesian(self.plot_radius, end_theta),
            ]
            color = self.edge_colors[i]
            codes = [Path.MOVETO, Path.CURVE3, Path.CURVE3]
            lw = self.edge_widths[i]
            path = Path(verts, codes)
            patch = patches.PathPatch(
                path, lw=lw, edgecolor=color, zorder=1, **self.edgeprops
            )
            self.ax.add_patch(patch)


    def draw_group_labels(self):
        """
        Renders group labels to the figure.
        """
        for i, label in enumerate(self.groups):
            label_x = self.group_label_coords["x"][i]
            label_y = self.group_label_coords["y"][i]
            label_ha = self.group_label_aligns["has"][i]
            label_va = self.group_label_aligns["vas"][i]
            color = self.group_label_color[i]
            self.ax.text(
                s=label,
                x=label_x,
                y=label_y,
                ha=label_ha,
                va=label_va,
                color=color,
                fontsize=self.fontsize,
                family=self.fontfamily,
            )


    def draw_legend(self):
        # Get the label and color for each group
        seen = set()

        # Gets colors in RGBA
        colors_group = [
            x for x in self.node_colors if not (x in seen or seen.add(x))
        ]

        # Gets group labels
        labels_group = sorted(
            set([self.graph.nodes[n][self.node_color] for n in self.nodes])
        )

        # Create patchList to use as handle for plt.legend()
        patchlist = []
        for color, label in zip(colors_group, labels_group):
            # Convert RGBA to HEX value
            color = to_hex(color, keep_alpha=True)
            data_key = patches.Patch(color=color, label=label)
            patchlist.append(data_key)

        # Set the labels with the custom patchList
        self.ax.legend(
            handles=patchlist,
            loc="lower center",
            # Half number of columns for total of labels for the
            # groups
            ncol=int(len(labels_group) / 2),
            bbox_to_anchor=(0.5, -0.05),
        )

        # Make legend handle accessible for manipulation outside of the plot
        self.legend_handles = patchlist