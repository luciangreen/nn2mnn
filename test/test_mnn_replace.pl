:- begin_tests(mnn_replace).

:- use_module('../mnn_replace').
:- use_module(library(http/json)).

test(mnn_evaluation_or) :-
    MNN = mnn([neuron(out, all, boolean(or))]),
    mnn_replace:mnn_eval(MNN, [0,1], 1).

test(observe_nn_pairs) :-
    observe_nn(model1, [[0,0],[1,1]], Pairs),
    assertion(Pairs == [[0,0]-0,[1,1]-1]).

test(synthesise_or_rule) :-
    Examples = [[0,0]-0,[0,1]-1,[1,0]-1,[1,1]-1],
    synthesise_mnn(Examples, mnn([neuron(out, all, boolean(or))])).

test(exact_compare_passes) :-
    retractall(mnn_replace:replacement_mode(_)),
    assertz(mnn_replace:replacement_mode(exact)),
    Examples = [[0,0]-0,[0,1]-1,[1,0]-1,[1,1]-1],
    synthesise_mnn(Examples, MNN),
    compare_nn_mnn(model1, MNN, Examples, report(Accuracy, _, Errors, _, _, _, _)),
    assertion(Accuracy =:= 1.0),
    assertion(Errors == []).

test(approximate_compare_passes_with_tolerance) :-
    retractall(mnn_replace:replacement_mode(_)),
    assertz(mnn_replace:replacement_mode(approximate(1.1))),
    MNN = mnn([neuron(out, all, linear([1,1], 0))]),
    TestSet = [[0,0]-0,[0,1]-1,[1,0]-1,[1,1]-1],
    compare_nn_mnn(model1, MNN, TestSet, report(_, _, Errors, _, _, _, _)),
    assertion(Errors == []).

test(counterexample_refinement) :-
    M0 = mnn([neuron(out, all, lookup([[0,0]-0], 0))]),
    once(refine_mnn(M0, [[0,1]-1], M1)),
    once(mnn_replace:mnn_eval(M1, [0,1], 1)).

test(counterexample_overrides_stale_lookup_value) :-
    M0 = mnn([neuron(out, all, lookup([[0,1]-0], 0))]),
    once(refine_mnn(M0, [[0,1]-1], M1)),
    once(mnn_replace:mnn_eval(M1, [0,1], 1)).

test(optimise_lookup_sorts) :-
    M0 = mnn([neuron(out, all, lookup([[1,1]-1,[0,0]-0], 0))]),
    optimise_mnn(M0, M1),
    M1 = mnn([neuron(out, all, lookup([[0,0]-0,[1,1]-1], 0))]).

test(candidate_generation) :-
    Examples = [[0,0]-0,[0,1]-1,[1,0]-1,[1,1]-1],
    mnn_candidates(Examples, Candidates),
    assertion(member(mnn([neuron(out, all, boolean(or))]), Candidates)),
    identify_candidates(model1, [[0,0],[0,1],[1,0],[1,1]],
                        [candidate(gate, _, _, _, evidence(threshold_like))]).

test(explanation_reports_evaluation) :-
    MNN = mnn([neuron(out, all, boolean(or))]),
    explain_mnn(MNN, [0,1], explanation(out, all, boolean(or), [0,1], 1)).

test(component_replacement) :-
    once(distil_nn(model1, gate, MNN)),
    once(replace_component(model1, gate, MNN, Hybrid)),
    once(nn_component_input(Hybrid, gate, [1,0], Out)),
    assertion(Out == 1).

test(multiple_replacements_component_lookup) :-
    MNN = mnn([neuron(out, all, boolean(or))]),
    Hybrid = hybrid(model1, [gate-MNN,full-MNN]),
    once(nn_component(Hybrid, gate)),
    once(nn_input(Hybrid, [0,1], 1)).

test(replacement_ratio_value) :-
    MNN = mnn([neuron(out, all, boolean(or))]),
    once(replace_component(model1, gate, MNN, Hybrid)),
    once(replacement_ratio(model1, Hybrid, Ratio)),
    assertion(Ratio =:= 1.0).

test(benchmark_generation) :-
    once(distil_nn(model1, gate, MNN)),
    once(mnn_benchmark(model1, MNN, benchmark(_, _, ReplacementAccuracy, _, _, _, _, _, _))),
    assertion(ReplacementAccuracy =:= 1.0).

test(xor_benchmark_uses_model_outputs_as_labels) :-
    once(distil_nn(model_xor, gate, MNN)),
    once(mnn_benchmark(model_xor, MNN,
                       benchmark(model_xor, OriginalAccuracy, ReplacementAccuracy,
                                 Agreement, _, _, _, _, _))),
    assertion(OriginalAccuracy =:= 1.0),
    assertion(ReplacementAccuracy =:= 1.0),
    assertion(Agreement =:= 1.0).

test(benchmark_json_is_valid_json) :-
    tmp_file_stream(text, Path, Out),
    once(mnn_replace:write_benchmark_json(
        Out, benchmark(model1, 1.0, 0.75, 0.75, 1, 2, 20, 10, 1.0))),
    close(Out),
    setup_call_cleanup(
        open(Path, read, In),
        ( json_read_dict(In, Dict),
          assertion(Dict.model == "model1"),
          assertion(Dict.replacement_accuracy =:= 0.75)
        ),
        close(In)),
    delete_file(Path).

test(full_replacement_classification) :-
    MNN = mnn([neuron(out, all, boolean(or))]),
    once(replace_component(model1, gate, MNN, Hybrid)),
    mnn_replace:classify_result(Hybrid, 'FULL REPLACEMENT').

test(integration_research_report) :-
    once(mnn_research(model1, report(ResultType, Verification, _Benchmark, Repls))),
    assertion(ResultType == 'FULL REPLACEMENT'),
    assertion(Verification \= unknown),
    assertion(Repls \= []).

test(xor_research_verifies_against_xor_labels) :-
    once(mnn_research(model_xor,
                      report('FULL REPLACEMENT', verified,
                             benchmark(model_xor, 1, 1, 1, _, _, _, _, _),
                             Replacements))),
    assertion(Replacements = [_]).

:- end_tests(mnn_replace).
