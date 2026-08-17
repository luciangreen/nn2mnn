# nn2mnn
Neural Net to Manual Neuronet

## Overview
This repository now includes a minimal SWI-Prolog NN→MNN research MVP in `mnn_replace.pl`.

Implemented core capabilities include:
- explicit inspectable MNN representation (`mnn([neuron(...)])`)
- NN abstraction predicates (`nn_load/1`, `nn_input/3`, `nn_component/2`, `nn_component_input/4`)
- behaviour observation and candidate identification
- MNN synthesis (`synthesise_mnn/2`, `mnn_candidates/2`) and distillation (`distil_nn/3`)
- component replacement into hybrid models
- exact/approximate comparison modes and verification
- counterexample-guided refinement and optimisation
- progressive replacement and replacement-ratio calculation
- benchmark and experiment artifact generation (`results/experiment-001/*`)
- top-level APIs (`mnn_replace/2`, `mnn_replace/3`, `mnn_benchmark/3`, `mnn_research/2`)

## Run tests
```bash
swipl -q -s test/test_mnn_replace.pl -g run_tests -t halt
```

## CLI examples
```bash
swipl mnn_cli.pl --model model1
swipl mnn_cli.pl --model model1 --component gate
swipl mnn_cli.pl --model model1 --progressive
swipl mnn_cli.pl --model model1 --benchmark
```

`--component` selects a named component (for the built-in demo models, `gate`).
