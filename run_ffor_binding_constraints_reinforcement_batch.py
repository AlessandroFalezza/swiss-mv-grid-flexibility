import argparse
import csv
import os
import subprocess
from pathlib import Path


GRID_NAME = "20_0"
YEARS = (2030, 2040, 2050)
MONTHS = range(1, 13)
HOURS = ("12:00:00", "19:00:00")
NOTEBOOK = Path("FFOR_definitivo_binding_constraints_reinforcement.ipynb")
OUTPUT_TAG = "ffor_reinforcement_data"


def target_time(month, hour):
    return f"{month:02d}-15 {hour}"


def output_dir(year, target):
    safe_target = target.replace(" ", "_").replace(":", "-")
    return Path("outputs") / f"{OUTPUT_TAG}_{GRID_NAME}_{year}_{safe_target}"


def output_complete(path):
    required = [
        "ffor_boundary_points.csv",
        "ffor_boundary_closed.csv",
        "ffor_polygon_delta_pq.png",
        "ffor_segment_search_log.csv",
        "line_flows_by_boundary_point.csv",
        "binding_summary_by_point.csv",
        "binding_constraints_by_point.csv",
        "binding_network_constraints_by_point.csv",
        "baseline_pf_summary.csv",
        "reinforcement_lines.csv",
    ]
    return all((path / name).exists() for name in required)


def parse_args():
    parser = argparse.ArgumentParser(
        description="Run the reinforcement FFOR notebook for the 15th day of every month at 12:00 and 19:00."
    )
    parser.add_argument("--years", nargs="+", type=int, default=list(YEARS))
    parser.add_argument("--hours", nargs="+", default=list(HOURS))
    parser.add_argument("--skip-existing", action="store_true")
    parser.add_argument(
        "--executed-notebooks-dir",
        default=str(Path("outputs") / "ffor_reinforcement_notebook_runs"),
        help="Directory where executed reinforcement notebook copies are saved.",
    )
    return parser.parse_args()


def run_one(year, month, hour, executed_notebooks_dir):
    target = target_time(month, hour)
    out_dir = output_dir(year, target)
    executed_notebooks_dir.mkdir(parents=True, exist_ok=True)
    executed_name = f"FFOR_definitivo_binding_constraints_reinforcement_{year}_{target.replace(' ', '_').replace(':', '-')}.ipynb"

    env = os.environ.copy()
    env["FFOR_GRID_NAME"] = GRID_NAME
    env["FFOR_DATA_YEAR"] = str(year)
    env["FFOR_TARGET_TIME"] = target
    env["FFOR_NEXT_TIME"] = target
    runtime_dir = Path(".jupyter_runtime").resolve()
    runtime_dir.mkdir(parents=True, exist_ok=True)
    env["JUPYTER_RUNTIME_DIR"] = str(runtime_dir)
    env["JUPYTER_ALLOW_INSECURE_WRITES"] = "1"

    cmd = [
        "jupyter",
        "nbconvert",
        "--to",
        "notebook",
        "--execute",
        str(NOTEBOOK),
        "--output",
        executed_name,
        "--output-dir",
        str(executed_notebooks_dir),
        "--ExecutePreprocessor.timeout=-1",
    ]
    print(f"Running reinforcement {year} {target} -> {out_dir}", flush=True)
    subprocess.run(cmd, check=True, env=env)
    stale_failure_marker = out_dir / "run_failed.txt"
    if stale_failure_marker.exists():
        stale_failure_marker.unlink()
    return out_dir


def main():
    args = parse_args()
    executed_notebooks_dir = Path(args.executed_notebooks_dir)
    completed = []
    skipped = []
    failures = []

    for year in args.years:
        for month in MONTHS:
            for hour in args.hours:
                target = target_time(month, hour)
                out_dir = output_dir(year, target)
                if args.skip_existing and output_complete(out_dir):
                    print(f"Skipping complete reinforcement output: {out_dir}", flush=True)
                    skipped.append(out_dir)
                    continue
                try:
                    completed.append(run_one(year, month, hour, executed_notebooks_dir))
                except subprocess.CalledProcessError as exc:
                    failure = {
                        "year": year,
                        "target_time": target,
                        "output_dir": str(out_dir),
                        "returncode": exc.returncode,
                        "message": str(exc),
                    }
                    failures.append(failure)
                    out_dir.mkdir(parents=True, exist_ok=True)
                    with (out_dir / "run_failed.txt").open("w", encoding="utf-8") as handle:
                        handle.write(f"Reinforcement FFOR run failed for {year} {target}\n")
                        handle.write(f"Return code: {exc.returncode}\n")
                        handle.write(f"{exc}\n")
                    print(f"FAILED reinforcement {year} {target}: {exc}", flush=True)

    print(f"Completed reinforcement runs: {len(completed)}", flush=True)
    print(f"Skipped complete reinforcement runs: {len(skipped)}", flush=True)
    print(f"Failed reinforcement runs: {len(failures)}", flush=True)

    failure_path = Path("outputs") / "ffor_reinforcement_batch_failures.csv"
    failure_path.parent.mkdir(parents=True, exist_ok=True)
    with failure_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=["year", "target_time", "output_dir", "returncode", "message"])
        writer.writeheader()
        if failures:
            writer.writerows(failures)
    if failures:
        print(f"Failure log: {failure_path}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
