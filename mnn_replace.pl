:- module(mnn_replace,
    [ mnn_replace/2,
      mnn_replace/3,
      mnn_benchmark/3,
      mnn_research/2,
      nn_load/1,
      nn_input/3,
      nn_component/2,
      nn_component_input/4,
      observe_nn/3,
      observe_nn_component/4,
      identify_candidates/3,
      synthesise_mnn/2,
      mnn_candidates/2,
      distil_nn/3,
      replace_component/4,
      compare_nn_mnn/4,
      verify_replacement/3,
      refine_mnn/3,
      optimise_mnn/2,
      optimise_until_fixed_point/2,
      find_mnn_replacements/2,
      progressive_replace/2,
      replacement_ratio/3,
      explain_mnn/3,
      run_cli/1
    ]).

:- use_module(library(lists)).
:- use_module(library(apply)).
:- use_module(library(pairs)).
:- use_module(library(filesex)).
:- use_module(library(readutil)).
:- use_module(library(statistics)).

:- dynamic replacement_mode/1.

replacement_mode(exact).

% ------------------------------------------------------------------
% Minimal built-in NN fixtures.
% ------------------------------------------------------------------

nn_load(model1).
nn_load(model_xor).

nn_input(model1, [A,B], Out) :-
    ((A =:= 1 ; B =:= 1) -> Out = 1 ; Out = 0).
nn_input(model_xor, [A,B], Out) :-
    (A =:= B -> Out = 0 ; Out = 1).
nn_input(hybrid(NN, Replacements), Input, Output) :-
    ( memberchk(full-MNN, Replacements)
    -> mnn_eval(MNN, Input, Output)
    ; nn_input(NN, Input, Output)
    ).

nn_component(model1, gate).
nn_component(model_xor, gate).
nn_component(hybrid(NN, Replacements), Component) :-
    ( member(Component-_, Replacements)
    ; nn_component(NN, Component)
    ).

nn_component_input(model1, gate, Input, Output) :-
    nn_input(model1, Input, Output).
nn_component_input(model_xor, gate, Input, Output) :-
    nn_input(model_xor, Input, Output).
nn_component_input(hybrid(NN, Replacements), Component, Input, Output) :-
    ( memberchk(Component-MNN, Replacements)
    -> mnn_eval(MNN, Input, Output)
    ; nn_component_input(NN, Component, Input, Output)
    ).

% ------------------------------------------------------------------
% Observation and candidate identification.
% ------------------------------------------------------------------

observe_nn(Model, Inputs, Pairs) :-
    maplist(observe_input(Model), Inputs, Pairs).

observe_input(Model, Input, Input-Output) :-
    nn_input(Model, Input, Output).

observe_nn_component(Model, Component, Inputs, Pairs) :-
    maplist(observe_component(Model, Component), Inputs, Pairs).

observe_component(Model, Component, Input, Input-Output) :-
    nn_component_input(Model, Component, Input, Output).

identify_candidates(Model, Inputs, Candidates) :-
    findall(candidate(Component, Inputs, Outputs, Complexity, Evidence),
        ( nn_component(Model, Component),
          observe_nn_component(Model, Component, Inputs, Pairs),
          pairs_values(Pairs, Outputs),
          complexity_score(Pairs, Complexity),
          candidate_evidence(Pairs, Evidence)
        ),
        Candidates).

complexity_score(Pairs, Complexity) :-
    pairs_values(Pairs, Outputs),
    sort(Outputs, U),
    length(U, Distinct),
    length(Pairs, Total),
    (Total =:= 0 -> Complexity = 0 ; Complexity is Distinct / Total).

candidate_evidence(Pairs, evidence(Type)) :-
    ( detects_truth_table(Pairs, or)  -> Type = threshold_like
    ; detects_truth_table(Pairs, and) -> Type = boolean_relationship
    ; detects_truth_table(Pairs, xor) -> Type = decision_tree_like
    ; Type = general_pattern
    ).

% ------------------------------------------------------------------
% MNN synthesis/evaluation/explanations.
% ------------------------------------------------------------------

synthesise_mnn(Examples, mnn([neuron(out, all, Rule)])) :-
    choose_rule(Examples, Rule).

mnn_candidates(Examples, Candidates) :-
    findall(mnn([neuron(out, all, Rule)]), candidate_rule(Examples, Rule), Raw),
    sort(Raw, Candidates).

choose_rule(Examples, Rule) :-
    candidate_rule(Examples, Rule), !.
choose_rule(Examples, lookup(Examples, 0)).

candidate_rule(Examples, boolean(or))  :- detects_truth_table(Examples, or).
candidate_rule(Examples, boolean(and)) :- detects_truth_table(Examples, and).
candidate_rule(Examples, boolean(xor)) :- detects_truth_table(Examples, xor).
candidate_rule(Examples, lookup(Examples, 0)).

mnn_eval(mnn([neuron(_, _, Rule)]), Input, Output) :-
    eval_rule(Rule, Input, Output).

eval_rule(boolean(or), Input, Output) :-
    (member(X, Input), X =:= 1 -> Output = 1 ; Output = 0).
eval_rule(boolean(and), Input, Output) :-
    (forall(member(X, Input), X =:= 1) -> Output = 1 ; Output = 0).
eval_rule(boolean(xor), [A,B], Output) :-
    (A =:= B -> Output = 0 ; Output = 1).
eval_rule(threshold(Weights, Threshold), Input, Output) :-
    dot(Weights, Input, Sum),
    (Sum >= Threshold -> Output = 1 ; Output = 0).
eval_rule(linear(Weights, Bias), Input, Output) :-
    dot(Weights, Input, Sum),
    Output is Sum + Bias.
eval_rule(lookup(Pairs, Default), Input, Output) :-
    ( memberchk(Input-Output0, Pairs)
    -> Output = Output0
    ; Output = Default
    ).

dot(Ws, Xs, Sum) :-
    maplist(product, Ws, Xs, Products),
    sum_list(Products, Sum).

product(A, B, P) :- P is A * B.

explain_mnn(mnn([neuron(Id, Inputs, Rule)]), Input, explanation(Id, Inputs, Rule, Input, Output)) :-
    eval_rule(Rule, Input, Output).

detects_truth_table(Examples, Op) :-
    Required = [[0,0],[0,1],[1,0],[1,1]],
    forall(member(Input, Required),
        ( expected_output(Op, Input, Out),
          memberchk(Input-Out, Examples)
        )).

expected_output(or, [A,B], Out)  :- (A =:= 1 ; B =:= 1 -> Out = 1 ; Out = 0).
expected_output(and,[A,B], Out)  :- (A =:= 1, B =:= 1 -> Out = 1 ; Out = 0).
expected_output(xor,[A,B], Out)  :- (A =:= B -> Out = 0 ; Out = 1).

% ------------------------------------------------------------------
% Distillation, replacement, comparison and verification.
% ------------------------------------------------------------------

distil_nn(NN, Component, MNN) :-
    default_inputs(2, Inputs),
    observe_nn_component(NN, Component, Inputs, Examples),
    synthesise_mnn(Examples, MNN).

replace_component(NN, Component, MNN, hybrid(NN, [Component-MNN])).

compare_nn_mnn(NN, MNN, TestSet, report(Accuracy, Agreement, Errors, NNTime, MNNTime, NNMemory, MNNMemory)) :-
    timed_eval(nn_outputs(NN, TestSet, NNO), NNTime),
    timed_eval(mnn_outputs(MNN, TestSet, MNNO), MNNTime),
    compare_outputs(TestSet, NNO, MNNO, Accuracy, Agreement, Errors),
    term_size(NN, NNMemory),
    term_size(MNN, MNNMemory).

nn_outputs(NN, TestSet, Outputs) :-
    maplist(nn_output(NN), TestSet, Outputs).

nn_output(NN, Input-_, Out) :- !, nn_input(NN, Input, Out).
nn_output(NN, Input, Out) :- nn_input(NN, Input, Out).

mnn_outputs(MNN, TestSet, Outputs) :-
    maplist(mnn_output(MNN), TestSet, Outputs).

mnn_output(MNN, Input-_, Out) :- !, mnn_eval(MNN, Input, Out).
mnn_output(MNN, Input, Out) :- mnn_eval(MNN, Input, Out).

timed_eval(Goal, TimeMs) :-
    statistics(runtime, [T0|_]),
    call(Goal),
    statistics(runtime, [T1|_]),
    TimeMs is T1 - T0.

compare_outputs(TestSet, NNOut, MNNOut, Accuracy, Agreement, Errors) :-
    pairs_keys_values(Zipped, TestSet, NNOut, MNNOut),
    findall(E, (member(item(Input, N, M), Zipped), mismatch(Input, N, M, E)), Errors),
    length(Zipped, Total),
    length(Errors, ErrCount),
    ( Total =:= 0
    -> Accuracy = 0.0, Agreement = 0.0
    ; Correct is Total - ErrCount,
      Accuracy is Correct / Total,
      Agreement = Accuracy
    ).

pairs_keys_values([], [], [], []).
pairs_keys_values([item(Input, N, M)|T], [Input|I], [N|Ns], [M|Ms]) :-
    pairs_keys_values(T, I, Ns, Ms).

mismatch(Input-Expected, N, M, error(Input, expected(Expected), nn(N), mnn(M))) :-
    nonvar(Expected),
    !,
    mode_mismatch(Expected, M).
mismatch(Input, N, M, error(Input, nn(N), mnn(M))) :-
    mode_mismatch(N, M).

mode_mismatch(A, B) :-
    replacement_mode(exact),
    !,
    A =\= B.
mode_mismatch(A, B) :-
    replacement_mode(approximate(Tol)),
    Diff is abs(A-B),
    Diff > Tol.

verify_replacement(comparison(NN, MNN, TestSet), _Replacement, Result) :-
    compare_nn_mnn(NN, MNN, TestSet, report(Accuracy, _, Errors, _, _, _, _)),
    ( Accuracy =:= 1.0 -> Result = verified
    ; Accuracy >= 0.95 -> Result = empirically_equivalent(Accuracy)
    ; Errors \= [] -> Result = failed(Errors)
    ; Result = unknown
    ).
verify_replacement(bounded(Domain, NN, MNN), _Replacement, Result) :-
    compare_nn_mnn(NN, MNN, Domain, report(Accuracy, _, _, _, _, _, _)),
    (Accuracy =:= 1.0 -> Result = verified_bounded(Domain) ; Result = unknown).
verify_replacement(_, _, unknown).

% ------------------------------------------------------------------
% Refinement and optimisation.
% ------------------------------------------------------------------

refine_mnn(mnn([neuron(Id, Inputs, lookup(Pairs, Default))]), CounterExamples,
           mnn([neuron(Id, Inputs, lookup(NewPairs, Default))])) :-
    append(CounterExamples, Pairs, Combined),
    sort(Combined, NewPairs).
refine_mnn(MNN, _CounterExamples, MNN).

optimise_mnn(mnn([neuron(Id, Inputs, lookup(Pairs, Default))]),
             mnn([neuron(Id, Inputs, lookup(Sorted, Default))])) :-
    sort(Pairs, Sorted), !.
optimise_mnn(MNN, MNN).

optimise_until_fixed_point(MNN0, MNN) :-
    optimise_mnn(MNN0, MNN1),
    ( MNN1 == MNN0 -> MNN = MNN0 ; optimise_until_fixed_point(MNN1, MNN) ).

% ------------------------------------------------------------------
% Replacement search and progressive replacement.
% ------------------------------------------------------------------

find_mnn_replacements(NN, Candidates) :-
    default_inputs(2, Inputs),
    identify_candidates(NN, Inputs, RawCandidates),
    findall(replacement(Component, MNN, Agreement, Speedup, SizeReduction, Interpretability),
        ( member(candidate(Component, _, _, _, _), RawCandidates),
          distil_nn(NN, Component, MNN),
          make_testset(Inputs, TestSet),
          compare_nn_mnn(NN, MNN, TestSet, report(_, Agreement, _, NNTime, MNNTime, NNSize, MNNSize)),
          speedup(NNTime, MNNTime, Speedup),
          size_reduction(NNSize, MNNSize, SizeReduction),
          interpretability_score(MNN, Interpretability)
        ),
        Unsorted),
    sort(3, @>=, Unsorted, Candidates).

progressive_replace(NN, BestModel) :-
    find_mnn_replacements(NN, Repls),
    progressive_replace_(NN, Repls, NN, BestModel).

progressive_replace_(Current, [], _BestSoFar, Current).
progressive_replace_(Current, [replacement(Component, MNN, Agreement, _, _, _)|Rest], BestSoFar, Best) :-
    ( Agreement >= 0.95
    -> replace_component(Current, Component, MNN, Hybrid),
       progressive_replace_(Hybrid, Rest, Hybrid, Best)
    ; progressive_replace_(Current, Rest, BestSoFar, Best)
    ).

replacement_ratio(OriginalNN, hybrid(_, Replacements), Ratio) :-
    findall(C, nn_component(OriginalNN, C), Components0),
    sort(Components0, Components),
    length(Components, Total),
    findall(C, member(C-_, Replacements), Rs0),
    sort(Rs0, Rs),
    length(Rs, Replaced),
    ( Total =:= 0 -> Ratio = 0.0 ; Ratio is Replaced / Total ).
replacement_ratio(_, _, 0.0).

speedup(NNTime, MNNTime, Speedup) :-
    ( MNNTime =:= 0 -> Speedup = inf ; Speedup is NNTime / MNNTime ).

size_reduction(NNSize, MNNSize, Reduction) :-
    ( NNSize =:= 0 -> Reduction = 0.0 ; Reduction is (NNSize - MNNSize) / NNSize ).

interpretability_score(mnn([neuron(_, _, lookup(_, _))]), 0.8).
interpretability_score(mnn([neuron(_, _, boolean(_))]), 0.9).
interpretability_score(_, 0.5).

% ------------------------------------------------------------------
% Top-level API and benchmarking/report output.
% ------------------------------------------------------------------

mnn_replace(NN, Hybrid) :-
    mnn_replace(NN, [], Hybrid).

mnn_replace(NN, Options, Hybrid) :-
    set_mode_from_options(Options),
    ( memberchk(progressive(true), Options)
    -> progressive_replace(NN, Hybrid)
    ; memberchk(component(Component), Options)
    -> distil_nn(NN, Component, MNN),
       replace_component(NN, Component, MNN, Hybrid)
    ; nn_component(NN, Component),
      distil_nn(NN, Component, MNN),
      replace_component(NN, Component, MNN, Hybrid)
    ).

mnn_benchmark(NN, MNN, benchmark(or_problem, OriginalAccuracy, ReplacementAccuracy, Agreement,
                                 OriginalRuntime, ReplacementRuntime, OriginalSize, ReplacementSize,
                                 ReplacementRatio)) :-
    default_inputs(2, Inputs),
    make_testset(Inputs, TestSet),
    compare_nn_mnn(NN, MNN, TestSet,
                   report(Agreement, Agreement, _Errors,
                          OriginalRuntime, ReplacementRuntime,
                          OriginalSize, ReplacementSize)),
    OriginalAccuracy = 1.0,
    ReplacementAccuracy = Agreement,
    replace_component(NN, gate, MNN, Hybrid),
    replacement_ratio(NN, Hybrid, ReplacementRatio).

mnn_research(NN, report(ResultType, Verification, Benchmark, Replacements)) :-
    find_mnn_replacements(NN, Replacements),
    ( Replacements = [replacement(Component, MNN, _, _, _, _)|_]
    -> replace_component(NN, Component, MNN, Hybrid),
       make_testset([[0,0],[0,1],[1,0],[1,1]], TestSet),
       verify_replacement(comparison(NN, MNN, TestSet), Hybrid, Verification),
       mnn_benchmark(NN, MNN, Benchmark),
       classify_result(Hybrid, ResultType),
       write_experiment_files(NN, Component, MNN, Benchmark, Verification)
    ; ResultType = 'NO REPLACEMENT',
      Verification = unknown,
      Benchmark = none
    ).

classify_result(hybrid(_, Replacements), 'FULL REPLACEMENT') :-
    memberchk(full-_, Replacements), !.
classify_result(hybrid(_, [_|_]), 'PARTIAL REPLACEMENT') :- !.
classify_result(_, 'NO REPLACEMENT').

write_experiment_files(NN, Component, MNN, Benchmark, Verification) :-
    Dir = 'results/experiment-001',
    make_directory_path(Dir),
    atomic_list_concat([Dir, '/original-model.txt'], OM),
    atomic_list_concat([Dir, '/observations.pl'], Obs),
    atomic_list_concat([Dir, '/mnn.pl'], MnnF),
    atomic_list_concat([Dir, '/counterexamples.pl'], Cex),
    atomic_list_concat([Dir, '/benchmark.json'], BJ),
    atomic_list_concat([Dir, '/benchmark.csv'], BC),
    atomic_list_concat([Dir, '/report.md'], RM),
    setup_call_cleanup(open(OM, write, S1), writeq(S1, NN), close(S1)),
    default_inputs(2, Inputs),
    observe_nn(NN, Inputs, Pairs),
    setup_call_cleanup(open(Obs, write, S2), writeq(S2, observations(Pairs)), close(S2)),
    setup_call_cleanup(open(MnnF, write, S3), writeq(S3, MNN), close(S3)),
    setup_call_cleanup(open(Cex, write, S4), writeq(S4, counterexamples([])), close(S4)),
    setup_call_cleanup(open(BJ, write, S5), format(S5, '{"benchmark":"~q"}\n', [Benchmark]), close(S5)),
    setup_call_cleanup(open(BC, write, S6), format(S6, 'benchmark\n~q\n', [Benchmark]), close(S6)),
    setup_call_cleanup(open(RM, write, S7),
        format(S7, '~w\nEvidence: ~q\n', [Verification, Verification]),
        close(S7)),
    Component \= none.

default_inputs(2, [[0,0],[0,1],[1,0],[1,1]]).
default_inputs(N, Inputs) :-
    N > 0,
    findall(Bits, bits_of_length(N, Bits), Inputs).

bits_of_length(0, []).
bits_of_length(N, [B|Bs]) :-
    N > 0,
    member(B, [0,1]),
    N1 is N - 1,
    bits_of_length(N1, Bs).

make_testset(Inputs, TestSet) :-
    maplist(as_test_example, Inputs, TestSet).

as_test_example(Input, Input-_).

set_mode_from_options(Options) :-
    retractall(replacement_mode(_)),
    ( memberchk(mode(Mode), Options) -> assertz(replacement_mode(Mode)) ; assertz(replacement_mode(exact)) ).

% ------------------------------------------------------------------
% CLI entry point.
% ------------------------------------------------------------------

run_cli(Argv) :-
    parse_cli(Argv, Model, Options),
    ( memberchk(benchmark(true), Options)
    -> distil_nn(Model, gate, MNN),
       mnn_benchmark(Model, MNN, Report),
       writeln(Report)
    ; mnn_replace(Model, Options, Hybrid),
      writeln(Hybrid)
    ).

parse_cli(Argv, Model, Options) :-
    ( append(_, ['--model', ModelAtom|_], Argv) -> atom_string(Model, ModelAtom) ; Model = model1 ),
    findall(O, cli_option(Argv, O), Options).

cli_option(Argv, progressive(true)) :- member('--progressive', Argv).
cli_option(Argv, benchmark(true))   :- member('--benchmark', Argv).
cli_option(Argv, component(gate))   :- member('--component', Argv).

:- initialization(main, main).

main(Argv) :-
    ( Argv == []
    -> true
    ; run_cli(Argv)
    ).
