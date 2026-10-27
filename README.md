# nn2mnn

`nn2mnn` is a SWI-Prolog research MVP for replacing a neural-network-like
component with an inspectable manual neuron network (MNN). It currently ships
with two built-in Boolean models, rather than an importer for trained neural
network files.

## Requirements and setup

- SWI-Prolog 8.4 or newer (tested with SWI-Prolog 9.x)
- No third-party Prolog packages are required

Clone the repository and run commands from its root directory.

## Run the tests

Run the complete PlUnit suite:

```sh
swipl -q -s test/test_mnn_replace.pl -g run_tests -t halt
```

The tests cover MNN evaluation and synthesis, model observations and
candidate identification, exact and approximate comparison, lookup
counterexample refinement, optimisation, explanations, component and
progressive replacement, replacement ratios, benchmarking, JSON artifact
format, and the end-to-end research report.

## CLI

```sh
# Replace the built-in OR gate in model1
swipl mnn_cli.pl --model model1

# Select a component explicitly
swipl mnn_cli.pl --model model1 --component gate

# Replace eligible components progressively
swipl mnn_cli.pl --model model1 --progressive

# Print a benchmark instead of a replacement model
swipl mnn_cli.pl --model model_xor --benchmark
```

The available models are `model1` (Boolean OR) and `model_xor` (Boolean XOR).
Both expose one component named `gate`. With no `--model`, the CLI uses
`model1`. `--progressive` and `--benchmark` are flags; `--component` and
`--model` each take a value.

## Prolog API

Load the module with `:- use_module(mnn_replace).`. The exported predicates
are:

| Predicate | Behavior |
| --- | --- |
| `nn_load/1` | Succeeds for the built-in models `model1` and `model_xor`. |
| `nn_input/3` | Evaluates a model or a `hybrid/2` model on an input list. |
| `nn_component/2` | Enumerates a model's replaceable components. |
| `nn_component_input/4` | Evaluates one component, honoring an existing replacement. |
| `observe_nn/3`, `observe_nn_component/4` | Produce `Input-Output` example pairs for a model or component. |
| `identify_candidates/3` | Observes components and returns `candidate(Component, Inputs, Outputs, Complexity, Evidence)` terms. |
| `synthesise_mnn/2` | Synthesizes a Boolean OR/AND/XOR rule when the examples match its full 2-bit truth table; otherwise creates a lookup rule. |
| `mnn_candidates/2` | Returns the distinct Boolean-rule and lookup candidates consistent with the examples. |
| `distil_nn/3` | Observes the selected component over all four 2-bit Boolean inputs and synthesizes an MNN. |
| `replace_component/4` | Constructs `hybrid(OriginalModel, [Component-MNN])`. |
| `compare_nn_mnn/4` | Compares outputs and returns `report(Accuracy, Agreement, Errors, NNTimeMs, MNNTimeMs, NNTermSize, MNNTermSize)`. |
| `verify_replacement/3` | Checks agreement on a supplied comparison set or bounded domain. This is finite-set verification, not a proof over arbitrary inputs. |
| `refine_mnn/3` | Adds lookup counterexamples; a new value for an existing input overrides its old value. Other rule types are unchanged. |
| `optimise_mnn/2`, `optimise_until_fixed_point/2` | Sort lookup entries; other rules are unchanged. |
| `find_mnn_replacements/2` | Ranks component replacement candidates by agreement. |
| `progressive_replace/2` | Applies candidates with at least 0.95 agreement. |
| `replacement_ratio/3` | Reports the fraction of original components present in a hybrid's replacement list. |
| `explain_mnn/3` | Returns the selected neuron, rule, input, and evaluated output. |
| `mnn_replace/2`, `mnn_replace/3` | Replaces the first available component, or uses options `component(Name)`, `progressive(true)`, and `mode(exact)` / `mode(approximate(Tolerance))`. |
| `mnn_benchmark/3` | Measures accuracy, agreement, runtime, term size, and replacement ratio on the model's 2-bit input domain. |
| `mnn_research/2` | Runs candidate search, replacement, verification, benchmarking, and artifact writing. |
| `run_cli/1` | Parses and executes the CLI arguments. |

MNN terms have the form `mnn([neuron(Id, Inputs, Rule)])`. Supported rules are
`boolean(or)`, `boolean(and)`, `boolean(xor)`, `threshold(Weights, Threshold)`,
`linear(Weights, Bias)`, and `lookup(Pairs, Default)`. MNN evaluation is
available through `nn_input/3` on a hybrid model; `mnn_eval/3` is an internal
predicate and is not part of the exported API.

Exact comparison is the default. Approximate mode accepts outputs whose
absolute difference is no greater than the supplied tolerance. The mode is
module-wide state used by comparison predicates; `mnn_replace/3` resets it
from its options on each call.

## Research artifacts

`mnn_research/2` writes one directory per run under `results/`, starting with
`results/experiment-001/`. The directory is ignored by Git and contains:

- `original-model.txt` — quoted Prolog model term
- `observations.pl` — observed input/output pairs
- `mnn.pl` — synthesized MNN term
- `counterexamples.pl` — counterexample term (currently an empty list; discovery is not automatic)
- `benchmark.json` — JSON object with model, accuracy, agreement, runtime,
  term-size, and replacement-ratio fields
- `benchmark.csv` — the same benchmark fields in tabular form
- `report.md` — verification evidence and benchmark summary

Experiment numbering selects the first unused directory up to
`experiment-9999`.

## Current implementation limits / remaining work

- There is no importer or adapter for external neural-network formats or
  trained model weights; only the two built-in fixtures can be evaluated.
- Models currently expose one `gate` component and use a fixed two-input,
  Boolean observation domain. General component boundaries, input dimensions,
  and non-Boolean domains remain to be implemented.
- Synthesis recognizes full OR/AND/XOR truth tables or falls back to a lookup
  table. It does not learn arbitrary weighted neurons, simplify general
  functions, or produce a multi-neuron network.
- Refinement accepts supplied counterexamples but does not search for them;
  formal equivalence checking and automatic counterexample generation are
  future work. A `verified` result means the supplied finite test set agrees.
- Runtime values use coarse process CPU milliseconds and can be zero for tiny
  examples; term size is a structural estimate, not measured memory usage.
- CLI parsing currently has no help command or strict validation for unknown
  options.
