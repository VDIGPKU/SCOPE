# Setup

Run the commands below from the project root. Use Python 3.12, a CUDA-capable PyTorch installation, and EGL libraries for headless rendering.

## Python and source dependencies

```bash
python3.12 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip 'setuptools<81' wheel
python scripts/download_sources.py
python -m pip install -c constraints.txt -e '.[libero,services]'
python -m pip install -c constraints.txt -e external/LIBERO-PRO -e external/robosuite --no-deps
python -m pip install -c constraints.txt -e external/jaxls -e external/pyroki -e external/sam3
python -m pip install -c constraints.txt -e external/contact_graspnet_pytorch --no-build-isolation
patch -d external/contact_graspnet_pytorch -p1 < patches/contact_graspnet_pytorch-compat.patch
```

The downloader uses fixed source revisions and saves them in `external/`. Robosuite is the simulation dependency of LIBERO-Pro.

## Assets and weights

- **LIBERO-Pro:** use the assets in the downloaded source tree. Download `bddl_files` and `init_files` from the [LIBERO-Pro dataset](https://huggingface.co/datasets/zhouxueyang/LIBERO-Pro) into `external/LIBERO-PRO/libero/libero/` if absent. The demo uses the `libero_goal_swap` suite.
- **SAM 3:** download `sam3.pt` from [facebook/sam3](https://huggingface.co/facebook/sam3) and set `SAM3_CHECKPOINT_PATH` in `.env`. Alternatively, set `HF_TOKEN` and the service downloads its checkpoint on first launch.
- **Contact-GraspNet:** the [upstream source](https://github.com/elchun/contact_graspnet_pytorch) includes `checkpoints/contact_graspnet/config.yaml` and `checkpoints/contact_graspnet/checkpoints/model.pt`. Set `CONTACT_GRASPNET_ROOT` to the absolute path of `external/contact_graspnet_pytorch`, and `CONTACT_GRASPNET_CHECKPOINT_DIR` to its `checkpoints/contact_graspnet/checkpoints` directory.
- **Panda robot description:** `robot_descriptions` downloads the robot model into its cache on first service launch.

Create the LIBERO path configuration:

```bash
python scripts/configure_libero.py
```

Copy the printed `LIBERO_CONFIG_PATH` assignment into `.env`. All paths in `.env` should be absolute. `SAM3_PORT`, `GRASPNET_PORT`, and `PYROKI_PORT` can stay empty to use ports 8114, 8115, and 8116.

## Run

Start services with the command in the [README](../README.md). Service logs are written to `logs/`; model initialization may take a few minutes. Use `--graspnet-gpu 1` to put Contact-GraspNet on a second GPU. PyRoKi runs on CPU.

```bash
python scripts/demo.py --env-file .env --gpu 0 --check
python scripts/demo.py --env-file .env --gpu 0
```

Use `--seed 52` to choose another demo seed, `--max-actions 16` for more search actions, and `--max-tokens` or `--reasoning-effort` to match your model. Each invocation uses one task and one seed. A search action is an execution, diagnosis, repair, or replay.
