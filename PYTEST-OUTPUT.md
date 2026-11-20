# Full pytest output

```text
============================= test session starts ==============================
platform linux -- Python 3.12.3, pytest-8.4.2, pluggy-1.7.0 -- /home/hatch/workspace/sidestep-push/sidestep/../venv/bin/python
cachedir: .pytest_cache
rootdir: /home/hatch/workspace/sidestep-push/sidestep
configfile: pyproject.toml
plugins: anyio-4.16.0, asyncio-1.5.0
asyncio: mode=Mode.AUTO, debug=False, asyncio_default_fixture_loop_scope=None, asyncio_default_test_loop_scope=function
collecting ... collected 35 items

tests/test_gateway.py::test_requires_bearer_token_and_health_is_public PASSED [  2%]
tests/test_gateway.py::test_refusal_is_forwarded_unchanged_and_recorded_once PASSED [  5%]
tests/test_gateway.py::test_non_refusal_response_is_transparent PASSED   [  8%]
tests/test_gateway.py::test_stream_is_forwarded_byte_for_byte_and_audited_afterward PASSED [ 11%]
tests/test_gateway.py::test_upstream_failure_is_not_retried_and_is_counted PASSED [ 14%]
tests/test_gateway.py::test_models_endpoint_forwards_to_single_upstream PASSED [ 17%]
tests/test_gateway.py::test_layered_detection_signals PASSED             [ 20%]
tests/test_gateway.py::test_verdict_cache_key_is_provider_and_model_scoped PASSED [ 22%]
tests/test_gateway.py::test_fixture_has_40_safety_framed_prompts_and_is_not_sent_upstream PASSED [ 25%]
tests/test_ladder.py::test_plan_puts_every_provider_verbatim_first PASSED [ 28%]
tests/test_ladder.py::test_plan_runs_mutations_against_lowest_risk_provider_only PASSED [ 31%]
tests/test_ladder.py::test_plan_omits_capability_gated_mutations PASSED  [ 34%]
tests/test_ladder.py::test_plan_ends_with_decompose PASSED               [ 37%]
tests/test_ladder.py::test_mutation_pass_is_the_documented_order PASSED  [ 40%]
tests/test_ladder.py::test_ladder_hits_each_provider_before_mutating PASSED [ 42%]
tests/test_ladder.py::test_routes_to_next_provider_by_ascending_risk PASSED [ 45%]
tests/test_ladder.py::test_format_lock_answer_is_unwrapped PASSED        [ 48%]
tests/test_ladder.py::test_longest_non_refusing_choice_is_selected PASSED [ 51%]
tests/test_ladder.py::test_streaming_refusal_buffer_is_discarded_before_retry PASSED [ 54%]
tests/test_ladder.py::test_json_body_on_streaming_request_is_not_read_as_empty PASSED [ 57%]
tests/test_ladder.py::test_http_400_advances_instead_of_terminating PASSED [ 60%]
tests/test_ladder.py::test_http_429_is_unavailable_and_advances PASSED   [ 62%]
tests/test_ladder.py::test_http_403_with_filter_marker_is_a_refusal PASSED [ 65%]
tests/test_ladder.py::test_max_attempts_stops_after_cap PASSED           [ 68%]
tests/test_ladder.py::test_max_attempts_of_one_makes_a_single_call PASSED [ 71%]
tests/test_ladder.py::test_max_cost_stops_after_cost_breach PASSED       [ 74%]
tests/test_ladder.py::test_deadline_stops_loop_and_returns_best_decompose_answer PASSED [ 77%]
tests/test_ladder.py::test_all_refused_returns_full_attempt_log PASSED   [ 80%]
tests/test_ladder.py::test_all_unavailable_reports_upstream_unavailable PASSED [ 82%]
tests/test_ladder.py::test_observe_only_env_forwards_refusal_without_retry PASSED [ 85%]
tests/test_ladder.py::test_missing_provider_keys_are_skipped_without_crash PASSED [ 88%]
tests/test_ladder.py::test_provider_without_key_is_skipped_and_next_provider_runs PASSED [ 91%]
tests/test_ladder.py::test_unset_observe_flag_defaults_to_ladder PASSED  [ 94%]
tests/test_ladder.py::test_observe_only_loads_registry_default_upstream PASSED [ 97%]
tests/test_ladder.py::test_missing_provider_keys_are_skipped_without_crashing PASSED [100%]

============================== 35 passed in 1.25s ==============================
```
