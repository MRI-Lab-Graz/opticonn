# Final fix wave report

Verified in main(): base_output = <args.output_dir>/optimize; output_dir = base_output. Findings confirmed.

1. Hook (cross_validation_bootstrap_optimizer.py): now run_variance_decomposition(Path(output_dir), Path(output_dir) / "optimization_results"). Nothing else touched; nesting/try/except/logging unchanged.
2. Replaced test_run_accepts_a_sweep_shaped_output_dir_directly with AST guard test_variance_decomposition_hook_passes_output_dir_itself_as_sweep_root (arg1 is output_dir or Path(output_dir); arg2 is <that> / "optimization_results").
   RED (bug reintroduced via sed, then fix restored from a saved copy):
   E  AssertionError: Path(output_dir) / 'optimize'
   E  assert False
   FAILED tests/test_variance_decomposition.py::test_variance_decomposition_hook_passes_output_dir_itself_as_sweep_root
   1 failed, 21 passed
   GREEN with fix: 22 passed.
3. compute_strata: one percent-style warning (count of total, up to 5 examples); unparseable keys excluded from between_session and between_subject (no pseudo-subject), still in tracking_noise/parameter. Test test_compute_strata_warns_and_excludes_unparseable_keys (RED before fix, GREEN after).
4. run() unlinks prior CSV/summary (missing_ok) first; write_decomposition unchanged. Test test_run_twice_does_not_duplicate_output (RED: 8 == 4 rows before fix).
5. run() warns when collect_sweep_matrices is empty, naming the dir. Test test_run_warns_when_no_sweep_matrices_found (RED before, GREEN after).
6. Known-ceiling comment added at vecs_all_reps; test imports merged to top of file, assertions untouched.

Full suite: source braingraph_pipeline/bin/activate; python3 -m pytest tests/ -q -> 119 passed (was 116).
