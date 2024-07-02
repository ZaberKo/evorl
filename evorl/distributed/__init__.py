from .comm import (
    psum, pmean, pmin, pmax, unpmap, tree_unpmap, tree_pmean,
    split_key_to_devices,
    is_dist_initialized, get_process_id, get_global_rank
)

from .gradients import agent_gradient_update

from .sharding import tree_device_put, tree_device_get

PMAP_AXIS_NAME = "P"

POP_AXIS_NAME = "POP"
