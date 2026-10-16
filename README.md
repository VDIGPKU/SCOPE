# SCOPE

SCOPE generates robot programs, diagnoses execution traces, and repairs programs using recorded experience. It connects a vision-language model to robot perception and control APIs, then uses observations from execution to guide subsequent program revisions.

The workflow begins with an observation of the scene and the task instruction. SCOPE generates an initial program, executes it in the simulator, and records API interactions and visual observations. These records support diagnosis, program repair, and offline replay as the search proceeds. Programs and execution results are saved together so that each revision can be inspected alongside the evidence that informed it.

The quick start below walks through environment setup, service startup, program execution, and inspection of the results.

## Setup

Use Linux, Python 3.12, and an NVIDIA GPU, with CUDA-capable PyTorch and EGL support for headless rendering. The [setup guide](docs/setup.md) covers creating the Python environment, installing dependencies, downloading benchmark assets and model weights, and configuring local paths.

The setup includes LIBERO-Pro simulation dependencies, SAM 3 for segmentation, Contact-GraspNet for grasp proposals, and PyRoKi for inverse kinematics. Source dependencies, benchmark data, and model weights are downloaded separately during setup. Run the commands below from the project root after creating the environment at `.venv/`.

Create a local environment configuration from the provided template:

```bash
cp .env.example .env
```

Fill in `OPENAI_BASE_URL`, `OPENAI_API_KEY`, and `SCOPE_AGENT_MODEL` for a vision-capable Chat Completions model. The model receives task instructions and visual execution evidence and produces the programs and diagnoses used by SCOPE. All values in the template are initially empty.

Set `LIBERO_CONFIG_PATH` to the directory containing the generated LIBERO configuration. Set `SAM3_CHECKPOINT_PATH`, `CONTACT_GRASPNET_ROOT`, and `CONTACT_GRASPNET_CHECKPOINT_DIR` to the corresponding downloaded files and directories, using the paths described in the setup guide. Paths in `.env` should be absolute. The service port variables can remain empty to use their defaults.

## Quick Start

Start the perception and inverse-kinematics services in one terminal:

```bash
.venv/bin/python scripts/start_services.py --env-file .env --gpu 0
```

This starts SAM 3 and Contact-GraspNet on the selected GPU and PyRoKi on CPU. Service logs are written to `logs/`. Keep this terminal running while using SCOPE, and allow the models to finish loading before starting the task.

In another terminal, start SCOPE:

```bash
.venv/bin/python scripts/demo.py --env-file .env --gpu 0
```

SCOPE loads the task, captures its initial observation, and generates the first program. It then executes and revises programs using the recorded observations and diagnostic feedback. The terminal reports progress through these steps as the run advances.

## Outputs

Programs, execution videos, diagnostics, and `summary.md` are saved in `outputs/demo/`. Start with `summary.md` to review the current search status and recorded attempts. The saved program files show the initial and subsequent revisions, while the execution videos and traces provide context for the corresponding outcomes.

Add `--resume` to the same command to continue a run from its saved state. To start a separate run, use a different output directory, such as `--output outputs/run-2`. The [setup guide](docs/setup.md) also describes the seed, search-budget, and model options.

