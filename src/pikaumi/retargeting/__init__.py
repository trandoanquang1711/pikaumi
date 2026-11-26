"""Multi-robot retargeting engines (`C-STRO`, `handumi_lm`, `dls_myopic`) & benchmark suite."""

from pikaumi.retargeting.cstro import (
    CStroConfig,
    build_cstro_problem,
    compute_contact_precision_multipliers,
    solve_episode_cstro,
)
from pikaumi.retargeting.pika_to_robot import (
    map_table_tcp_to_robot_ee,
    prepare_episode_robot_targets,
)

__all__ = [
    "CStroConfig",
    "build_cstro_problem",
    "compute_contact_precision_multipliers",
    "map_table_tcp_to_robot_ee",
    "prepare_episode_robot_targets",
    "solve_episode_cstro",
]
