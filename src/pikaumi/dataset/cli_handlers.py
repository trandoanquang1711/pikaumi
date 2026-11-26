"""CLI handlers for `pika dataset {...}` and `pika convert`."""

from __future__ import annotations

import argparse
from pathlib import Path

from pikaumi.dataset.curation import curate_dataset
from pikaumi.dataset.direction import analyze_dataset_direction
from pikaumi.dataset.merging import merge_datasets
from pikaumi.dataset.quality import validate_dataset
from pikaumi.dataset.umi_delta import export_umi_delta
from pikaumi.robots.registry import list_robot_names


def _expand_robots(raw_list: list[str]) -> list[str]:
    if len(raw_list) == 1 and raw_list[0] == "all":
        return list(list_robot_names())
    return list(raw_list)


def dispatch_dataset(args: argparse.Namespace) -> int:
    cmd = args.dataset_cmd
    if cmd == "validate":
        rep = validate_dataset(args.dataset_dir, quality_config_path=args.quality_config)
        print(
            f"[QA] validate {args.dataset_dir}: passed={rep['num_passed']}/{rep['num_episodes']} "
            f"(rejected={rep['num_rejected']})"
        )
        return 0 if rep["num_rejected"] == 0 else 1

    if cmd == "direction":
        rep = analyze_dataset_direction(args.dataset_dir)
        flagged = rep["flagged_reversed_episodes"]
        print(f"[QA] direction {args.dataset_dir}: flagged_reversed={flagged}")
        return 0 if not flagged else 1

    if cmd == "screen":
        from pikaumi.dataset.screening import screen_dataset_for_robots

        robots = _expand_robots(args.robots)
        reps = screen_dataset_for_robots(
            args.dataset_dir,
            robots=robots,
            solver=args.solver,
            retarget_mode=args.retarget_mode,
        )
        for rname, rep in reps.items():
            print(
                f"[SCREEN] {rname:<20} solver={args.solver:<10} "
                f"OK={rep['counts']['OK']} WARN={rep['counts']['WARN']} REJECT={rep['counts']['REJECT']}"
            )
        return 0

    if cmd == "qa":
        from pikaumi.dataset.screening import screen_dataset_for_robots

        q_rep = validate_dataset(args.dataset_dir)
        d_rep = analyze_dataset_direction(args.dataset_dir)
        robots = _expand_robots(args.robots)
        s_reps = screen_dataset_for_robots(args.dataset_dir, robots=robots, solver=args.solver)
        print(
            f"[QA] {args.dataset_dir}: validate={q_rep['num_passed']}/{q_rep['num_episodes']}, "
            f"reversed={d_rep['flagged_reversed_episodes']}"
        )
        for rname, s_rep in s_reps.items():
            print(
                f"     robot={rname:<18} OK={s_rep['counts']['OK']} "
                f"WARN={s_rep['counts']['WARN']} REJECT={s_rep['counts']['REJECT']}"
            )
        return 0

    if cmd == "curate":
        man = curate_dataset(
            args.dataset_dir,
            args.output_dir,
            robot=args.robot,
            exclude_episodes=list(args.exclude or []),
        )
        print(
            f"[OK] Curated {man['kept_episodes']} episode(s) -> {args.output_dir} "
            f"(dropped={len(man['dropped_episodes'])})"
        )
        return 0

    if cmd == "merge":
        prov = merge_datasets(args.sources, args.output_dir, task=args.task)
        print(f"[OK] Merged {prov['num_episodes']} episode(s) -> {args.output_dir}")
        return 0

    if cmd == "export":
        from pikaumi.dataset.external_layouts import export_external_follower_layout

        res = export_external_follower_layout(
            args.dataset_dir,
            layout=args.layout,
            output_dir=args.output_dir,
        )
        print(f"[OK] Exported {res['num_episodes']} episode(s) in layout={args.layout} -> {args.output_dir}")
        return 0

    raise ValueError(f"Unknown dataset subcommand: {cmd}")


def dispatch_convert(args: argparse.Namespace) -> int:
    if args.format == "umi-delta":
        info = export_umi_delta(
            args.dataset_dir,
            args.output_dir,
            allow_flagged_episodes=bool(getattr(args, "allow_flagged_episodes", False)),
        )
        print(
            f"[OK] Converted {info['num_episodes']} episode(s) to umi-delta (11D state / 14D action) -> {args.output_dir}"
        )
        return 0

    from pikaumi.dataset.canonical import convert_dataset_to_robot_joints

    info = convert_dataset_to_robot_joints(
        Path(args.dataset_dir),
        Path(args.output_dir),
        robot=args.robot,
        solver=args.solver,
        retarget_mode=args.retarget_mode,
        allow_flagged_episodes=args.allow_flagged_episodes,
    )
    print(
        f"[OK] Converted {info['num_episodes']} episode(s) for robot={args.robot} "
        f"(solver={args.solver}) -> {args.output_dir}"
    )
    return 0
