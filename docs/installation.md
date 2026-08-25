# Installation

swarp needs **Python ≥ 3.12**.
A CUDA GPU is optional: every kernel is compiled for the CPU too, and the test suite runs there.
The core install pulls in only `warp-lang`, `torch` and `numpy`.

## From a clone

```bash
git clone https://github.com/ddebenedittis/swarp.git && cd swarp
uv venv
uv pip install -e . --group dev
uv run pytest -m "not gpu"
```

The plain-pip equivalent, if you would rather not use [uv](https://docs.astral.sh/uv/):

```bash
python3.12 -m venv .venv && source .venv/bin/activate
pip install -e .
```

swarp is not on PyPI yet.

## Extras

Each extra is opt-in so the core stays lean.

| extra | install | what it adds |
|---|---|---|
| `viz` | `uv pip install -e '.[viz]'` | the pygame viewer, headless frames and mp4/webm export — see [Visualization](visualization.md) |
| `torchrl` | `uv pip install -e '.[torchrl]'` | the TorchRL `EnvBase` wrapper `swarp.interop.torchrl.SwarpEnv` |
| `bench` | `uv pip install -e '.[bench]'` | VMAS, for `python -m swarp.benchmark.compare_vmas` (it pins `numpy < 2`) |
| `bench-jaxmarl`, `bench-camar` | see [Benchmarks](benchmarks.md) | the JAX-based simulators the cross-simulator comparison drives, each in its own venv |
| `dev` (dependency group) | `uv pip install -e . --group dev` | pytest and ruff |
| `docs` (dependency group) | `uv sync --only-group docs` | Sphinx and the theme used to build this site |

The JAX-based benchmark extras genuinely cannot share one environment with `bench`: VMAS pins `numpy < 2` while JaxMARL and CAMAR want a recent numpy, so `compare_sims` drives one subprocess per simulator with that simulator's own interpreter.

## Verifying the install

```bash
uv run pytest -m "not gpu"     # what CI runs: dynamics, gradients, neighbors, collisions, determinism
uv run pytest                  # adds the CUDA-only tests
```

Three pytest markers are declared:

`gpu`
: needs a CUDA device — deselect on a CPU-only machine with `-m "not gpu"`.

`slow`
: large batches, long rollouts, `gradcheck`.

`viz`
: needs the `viz` extra; these tests skip cleanly without it.

A quick smoke check that the kernels compile and the batch really is on-device:

```python
import torch, swarp

env = swarp.make("navigation", n_envs=64, n_agents=4, device="cpu")
obs = env.reset()
obs, reward, done, info = env.step(torch.zeros(64, 4, env.act_dim))
print(obs.shape, reward.shape, done.shape)   # (64, 4, 19) (64, 4) (64,)
```

The first call compiles the Warp kernel modules and caches them under `~/.cache/warp/`, so it is noticeably slower than the ones that follow.

## GPU notes

Pass `device="cuda:0"` to run on the GPU; everything else is identical.
`Environment` defaults to `use_graph="auto"`, which captures the whole step into a CUDA graph whenever the device is CUDA and the scenario supplies a capturable hook — see [Performance](performance.md).

For a CPU-only environment (CI, a laptop without CUDA) install the CPU torch wheel to avoid downloading several GB of CUDA payload:

```bash
UV_TORCH_BACKEND=cpu uv pip install -e '.[viz]' --group dev
```

## The VMAS clone fallback

`swarp/benchmark/compare_vmas.py` falls back to a local `./VectorizedMultiAgentSimulator` checkout when `vmas` is not importable.
That directory is gitignored, is not a submodule, and is not part of the package — clone it yourself only if you want the comparison without installing the `bench` extra.
