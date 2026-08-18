:- begin_tests(mnn_replace).

:- use_module('../mnn_replace').

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

test(optimise_lookup_sorts) :-
    M0 = mnn([neuron(out, all, lookup([[1,1]-1,[0,0]-0], 0))]),
    optimise_mnn(M0, M1),
    M1 = mnn([neuron(out, all, lookup([[0,0]-0,[1,1]-1], 0))]).

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

test(integration_research_report) :-
    once(mnn_research(model1, report(ResultType, Verification, _Benchmark, Repls))),
    assertion(member(ResultType, ['PARTIAL REPLACEMENT','FULL REPLACEMENT','NO REPLACEMENT'])),
    assertion(Verification \= unknown),
    assertion(Repls \= []).

:- end_tests(mnn_replace).
