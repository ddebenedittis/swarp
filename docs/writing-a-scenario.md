# Writing a scenario

TODO: document the `FusedScenario` contract (the declarative buffer spec, the whole-step
hook, and the capture-safety rules for `launch_fused`).

Until this page exists, the README's
[Writing a scenario](../README.md#writing-a-scenario) section covers the plain
`Scenario` ABC — `make_world` / `reset_world` / `observation` / `agent_reward` — which is
all a torch-path scenario needs. The fused/CUDA-graph half of the contract is being
redesigned and is deliberately not documented here yet.
