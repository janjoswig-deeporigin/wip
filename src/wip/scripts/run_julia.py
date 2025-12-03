"""WIP Python CLI entry point to run registered Julia scripts with arguments"""

import argparse
import pathlib
import shlex
import subprocess

from loguru import logger


SCRIPT_PATH_MAP = {
    "prep": "simulation/prep.jl",
    "emin-bsm": "simulation/emin_bsm.jl",
}


def main():
    parser = argparse.ArgumentParser(
        description="Run a Julia script with specified arguments"
    )
    parser.add_argument(
        "script",
        help="Name of the Julia script to run (e.g., 'prep')",
    )
    parser.add_argument(
        "--taskset",
        help="Specify a taskset for CPU affinity; see also --taskset-args",
        action="store_true",
    )
    parser.add_argument(
        "--taskset-args",
        help="Arguments for taskset command",
        nargs="+",
        default="--cpu-list 0-4",
    )

    args, remainder = parser.parse_known_args()
    script_path = pathlib.Path(__file__).parent / SCRIPT_PATH_MAP[args.script]

    command = f"julia {script_path!s} " + " ".join(remainder)
    if args.taskset:
        command = "taskset " + " ".join(args.taskset_args) + " " + command
    command = shlex.split(command)
    logger.info(f"Running command: {' '.join(command)}")
    subprocess.run(command)


if __name__ == "__main__":
    main()