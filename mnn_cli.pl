:- use_module(mnn_replace).

:- initialization(main, main).

main(Argv) :-
    mnn_replace:run_cli(Argv).
