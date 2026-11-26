# Export Pika actions and LeRobot v2.1

The exporter writes the actual per-episode v2.1 format: one Parquet and one MP4 per camera per retained segment, plus `meta/{info.json,tasks.jsonl,episodes.jsonl,episodes_stats.jsonl,stats.json}`. It does not rename a v3.0 dataset or merely change its version metadata. See the [official format migration reference](https://huggingface.co/docs/lerobot/main/en/porting_datasets_v3).

Use the dataset CPU environment and explicitly choose v2.1:

```bash
export PIKA_PYTHON="$PWD/.venv-pika-dataset/bin/python"
export PIKA_VLA_PYTHON="$PWD/.venv-pika-dataset/bin/python"
./pika_data.sh lerobot 0 1 2 --output-root /path/to/fresh_export \
  --repo-id local/pick_and_place --format-version v2.1
```

This packages already prepared labels from `fresh_export/prepared/episodeN`. The output is `fresh_export/lerobot/` unless `--output-dir` is supplied. The default without `--format-version` remains v3.0. Existing output folders are never overwritten.

When the head was accidentally enabled during recording, exclude it before creating the source and labels:

```bash
./pika_data.sh export 0 --data-root /path/to/raw \
  --output-root /path/to/fresh_export --calibration-dir /path/to/calibration \
  --fps 30 --vista-only --exclude-head
./pika_data.sh prepare 0 --data-root /path/to/raw \
  --output-root /path/to/fresh_export --calibration-dir /path/to/calibration
```

The calibration directory must contain `{left,right}_fisheye_intrinsics.json` and `tcp/{left,right}_pika_tcp.json`. `--exclude-head` excludes its timing constraints as well as its images; raw files and capture metadata are not changed. Use a fresh source/label output to avoid reusing masks that were calculated with head synchronization.

State remains 11D: right TCP in left TCP, two rotation columns and two gripper widths. Action remains 14D: left then right, each `[dx,dy,dz,rx,ry,rz,next_width]`, TCP-local deltas in metres/radians and next absolute jaw width in metres. Every continuous valid run is an independent v2.1 episode. Training must group train/validation by original raw episode via `pika_source_reports.json`, not by the generated segment IDs.

Additional portable files:

- `actions/episodeN.npz`: screened action/state arrays, epoch timestamps, source rows and segment indices.
- `provenance/`: original camera timestamps and intrinsics, mapped to the retained label rows.
- `action_schema.json`: ordered action/state names and semantics.
- `meta/abs_action_norm_stats.json`: normalization statistics compatible with the local VLA loader.
- `validation.json`: v2.1 loader validation, exact action/state comparison and video boundary checks.

Read with `LeRobotDataset(repo_id, root=dataset_root, revision="v2.1", video_backend="pyav", use_delta_action=False)` and `use_corr_state=false`. Native RGB aspect ratio is preserved. Video uses H.264 at CRF 18; image statistics sample up to 100 frames/segment at spatial stride 16. Vector statistics use every retained row.

The operator requested combining raw episodes 0–148 and removing the head for the 2026-11-18 pick-and-place export. Existing numerical calibration was kept and the operator's acceptance recorded separately, preserving historical file flags for provenance. Quality screening thresholds were not relaxed.
