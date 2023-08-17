"""Contains the merge network class."""

from flow.networks.base import Network
from flow.core.params import InitialConfig
from flow.core.params import TrafficLightParams
from numpy import pi, sin, cos

INFLOW_EDGE_LEN = 100  # length of the inflow edges (needed for resets)
VEHICLE_LENGTH = 5

ADDITIONAL_NET_PARAMS = {
    # length of the merge edge
    "merge_length": 50,
    # length of the highway leading to the merge
    "pre_merge_length": 200,
    # length of the highway past the merge
    "post_merge_length": 100,
    # number of lanes in the merge
    "merge_lanes": 1,
    # number of lanes in the highway
    "highway_lanes": 1,
    # max speed limit of the network
    "speed_limit": 30,
}


class MergeNetwork(Network):
    """Network class for highways with a single in-merge.

    This network consists of a single or multi-lane highway network with an
    on-ramp with a variable number of lanes that can be used to generate
    periodic perturbation.

    Requires from net_params:

    * **merge_length** : length of the merge edge
    * **pre_merge_length** : length of the highway leading to the merge
    * **post_merge_length** : length of the highway past the merge
    * **merge_lanes** : number of lanes in the merge
    * **highway_lanes** : number of lanes in the highway
    * **speed_limit** : max speed limit of the network

    Usage
    -----
    >>> from flow.core.params import NetParams
    >>> from flow.core.params import VehicleParams
    >>> from flow.core.params import InitialConfig
    >>> from flow.networks import MergeNetwork
    >>>
    >>> network = MergeNetwork(
    >>>     name='merge',
    >>>     vehicles=VehicleParams(),
    >>>     net_params=NetParams(
    >>>         additional_params={
    >>>             'merge_length': 100,
    >>>             'pre_merge_length': 200,
    >>>             'post_merge_length': 100,
    >>>             'merge_lanes': 1,
    >>>             'highway_lanes': 1,
    >>>             'speed_limit': 30
    >>>         },
    >>>     )
    >>> )
    """

    def __init__(self,
                 name,
                 vehicles,
                 net_params,
                 initial_config=InitialConfig(),
                 traffic_lights=TrafficLightParams()):
        """Initialize a merge network."""
        for p in ADDITIONAL_NET_PARAMS.keys():
            if p not in net_params.additional_params:
                raise KeyError('Network parameter "{}" not supplied'.format(p))

        super().__init__(name, vehicles, net_params, initial_config,
                         traffic_lights)

    def specify_nodes(self, net_params):
        """See parent class."""
        angle = pi / 4
        merge = net_params.additional_params["merge_length"]
        premerge = net_params.additional_params["pre_merge_length"]
        postmerge = net_params.additional_params["post_merge_length"]

        nodes = [
            {
                "id": "inflow_highway",
                "x": -INFLOW_EDGE_LEN,
                "y": 0
            },
            {
                "id": "left",
                "y": 0,
                "x": 0
            },
            {
                "id": "top_merge",
                "y": 0,
                "x": premerge - 200,
                "radius": 10
            },
            {
                "id": "center",
                "y": 0,
                "x": premerge,
                "radius": 10
            },
            {
                "id": "right",
                "y": 0,
                "x": premerge + postmerge
            },
            {
                "id": "inflow_merge_bottom",
                "x": premerge - (merge + INFLOW_EDGE_LEN) * cos(angle),
                "y": -(merge + INFLOW_EDGE_LEN) * sin(angle)
            },
            {
                "id": "bottom",
                "x": premerge - merge * cos(angle),
                "y": -merge * sin(angle)
            },
            {
                "id": "inflow_merge_top",
                "x": premerge - 200 - (merge + INFLOW_EDGE_LEN) * cos(angle),
                "y": (merge + INFLOW_EDGE_LEN) * sin(angle)
            },
            {
                "id": "top",
                "x": premerge - 200 - merge * cos(angle),
                "y": merge * sin(angle)
            }
        ]

        return nodes

    def specify_edges(self, net_params):
        """See parent class."""
        merge = net_params.additional_params["merge_length"]
        premerge = net_params.additional_params["pre_merge_length"]
        postmerge = net_params.additional_params["post_merge_length"]

        edges = [
        {
            "id": "inflow_highway",
            "type": "highwayType",
            "from": "inflow_highway",
            "to": "left",
            "length": INFLOW_EDGE_LEN
        }, 
        {
            "id": "left",
            "type": "highwayType",
            "from": "left",
            "to": "top_merge",
            "length": premerge - 200
        }, 
        {
            "id": "top_merge",
            "type": "highwayType",
            "from": "top_merge",
            "to": "center",
            "length": premerge - 150
        },
        {
            "id": "inflow_merge_bottom",
            "type": "mergeType",
            "from": "inflow_merge_bottom",
            "to": "bottom",
            "length": INFLOW_EDGE_LEN
        }, 
        {
            "id": "bottom",
            "type": "mergeType",
            "from": "bottom",
            "to": "center",
            "length": merge
        }, 
        {
            "id": "inflow_merge_top",
            "type": "mergeType",
            "from": "inflow_merge_top",
            "to": "top",
            "length": INFLOW_EDGE_LEN
        },
        {
            "id": "top",
            "type": "mergeType",
            "from": "top",
            "to": "top_merge",
            "length": merge
        },
        {
            "id": "center",
            "type": "highwayType",
            "from": "center",
            "to": "right",
            "length": postmerge
        }]

        return edges

    def specify_types(self, net_params):
        """See parent class."""
        h_lanes = net_params.additional_params["highway_lanes"]
        m_lanes = net_params.additional_params["merge_lanes"]
        speed = net_params.additional_params["speed_limit"]

        types = [{
            "id": "highwayType",
            "numLanes": h_lanes,
            "speed": speed
        }, {
            "id": "mergeType",
            "numLanes": m_lanes,
            "speed": speed
        }]

        return types

    def specify_routes(self, net_params):
        """See parent class."""
        rts = {
            "inflow_highway": ["inflow_highway", "left", "top_merge", "center"],
            "left": ["left", "top_merge", "center"],
            "top_merge": ["top_merge", "center"],
            "center": ["center"],
            "inflow_merge_bottom": ["inflow_merge_bottom", "bottom", "center"],
            "bottom": ["bottom", "center"],
            "inflow_merge_top": ["inflow_merge_top", "top", "top_merge"],
            "top": ["top", "top_merge"],
        }

        return rts

    def specify_edge_starts(self):
        """See parent class."""
        premerge = self.net_params.additional_params["pre_merge_length"]
        postmerge = self.net_params.additional_params["post_merge_length"]

        edgestarts = [
            ("inflow_highway", 0), 
            ("left", INFLOW_EDGE_LEN + 0.1),
            ("top_merge", INFLOW_EDGE_LEN + premerge - 200 + 22.6),
            ("center", INFLOW_EDGE_LEN + premerge + 22.6),
            ("inflow_merge_bottom", INFLOW_EDGE_LEN + premerge + postmerge + 22.6),
            ("bottom", 2 * INFLOW_EDGE_LEN + premerge + postmerge + 22.7),
            ("inflow_merge_top", INFLOW_EDGE_LEN + premerge - 200 + postmerge + 22.6),
            ("top", 2 * INFLOW_EDGE_LEN + premerge - 200 + postmerge + 22.7),
        ]

        return edgestarts

    def specify_internal_edge_starts(self):
        """See parent class."""
        premerge = self.net_params.additional_params["pre_merge_length"]
        postmerge = self.net_params.additional_params["post_merge_length"]

        internal_edgestarts = [
            (":left", INFLOW_EDGE_LEN), 
            (":top_merge", INFLOW_EDGE_LEN + premerge - 200),
            (":center", INFLOW_EDGE_LEN + premerge + 0.1),
            (":bottom", 2 * INFLOW_EDGE_LEN + premerge + postmerge + 22.6),
            (":top", 2 * INFLOW_EDGE_LEN + premerge - 200 + postmerge + 22.6),
        ]

        return internal_edgestarts
