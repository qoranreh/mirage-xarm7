# Repository Guidelines

## Project Structure & Module Organization

The primary Python package lives in `mirage/mirage/`. Robosuite benchmark runners and YAML configurations are under `mirage/mirage/benchmark/robosuite`, while ROS 2 packages for Gazebo and custom messages are in `mirage/mirage/ros_ws/src`. `robosuite/`, `robomimic/`, and `mimicgen_environments/` are Git submodules; keep changes inside them intentional and record any updated submodule commit. Real-robot scripts are in `real_exps/`.

The version-controlled xArm integration is `xarm_extensions/mirage_xarm_state/`. The root `xarm_ws` symlink points to a machine-local build workspace and is not the durable source. Tests live beside their component, typically in `test/` or `tests/`. Do not commit generated `build/`, `install/`, `log/`, recordings, benchmark results, or model outputs.

## Build, Test, and Development Commands

```bash
git submodule update --init --recursive
python -m pip install -e ./mirage
```

These commands initialize dependencies and install Mirage in editable mode. Build the main ROS workspace after sourcing ROS 2 Humble:

```bash
source /opt/ros/humble/setup.bash
cd mirage/mirage/ros_ws && colcon build
```

Run a benchmark from `mirage/mirage/benchmark/robosuite` with `python3 run_robosuite_benchmark.py --config config/example_config.yaml`. Copy the example to the ignored `local_smoke.yaml` for machine-specific paths.

## Coding Style & Naming Conventions

Use four-space indentation and PEP 8 conventions: `snake_case` for modules, functions, topics, and parameters; `PascalCase` for classes; `UPPER_CASE` for constants. Add type hints where interfaces are not obvious. Robosuite uses Black and isort with a 120-character line limit (`robosuite/pyproject.toml`). Keep ROS parameters in YAML and make frames, topics, units, and timing assumptions explicit.

## Testing Guidelines

Use pytest and name files `test_*.py` and functions `test_*`. Run xArm tests from its package directory with `python3 -m pytest test` after sourcing the required ROS workspace. Run submodule suites from within the affected submodule. No repository-wide coverage threshold is configured; add deterministic regression tests for changed behavior and test fake or simulated hardware before a real robot.
## Remote-Sourced Code Modification Policy

Before modifying any code or file obtained from a remote repository, ask the user for explicit approval and describe the files to be changed and the intended modification. This includes cloned repositories, pulled files, Git submodules, vendored dependencies, copied upstream code, and patches based on upstream sources. Do not make the modification until the user has approved it.

After making an approved modification, record it in the repository-root `REMOTE_CHANGES.md` file. Each record must include:

* The date of the modification
* The source repository and upstream commit, tag, or version, when known
* The affected file paths
* The reason for the modification
* A concise summary of what was changed
* Any validation commands that were run

Create `REMOTE_CHANGES.md` if it does not already exist. For Git submodules, also record any change to the referenced submodule commit.


## Commit & Pull Request Guidelines

Recent commits use short, imperative, title-case summaries such as `Fix xArm state integration`. Keep each commit scoped to one subsystem. Pull requests should explain the behavior change, affected robots or environments, configuration changes, and exact validation commands. Link relevant issues; attach RViz screenshots, logs, or benchmark summaries when behavior is visual or experimental. Never commit credentials, robot IPs, local absolute paths, rosbag data, or checkpoints.
