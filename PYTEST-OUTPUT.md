# Full pytest output

```text
Cannot read termcap database;
using dumb terminal settings.
============================= test session starts ==============================
platform linux -- Python 3.12.1, pytest-8.4.2, pluggy-1.7.0 -- /home/.z/workspaces/con_PftXYeFvp6Lscz0i/.venv/bin/python
cachedir: .pytest_cache
rootdir: /home/workspace/sidestep-audit
configfile: pyproject.toml
testpaths: tests
plugins: anyio-4.16.0, asyncio-1.5.0
asyncio: mode=Mode.AUTO, debug=False, asyncio_default_fixture_loop_scope=None, asyncio_default_test_loop_scope=function
collecting ... collected 29 items

tests/test_gateway.py::test_requires_bearer_token_and_health_is_public PASSED [  3%]
tests/test_gateway.py::test_refusal_is_forwarded_unchanged_and_recorded_once PASSED [  6%]
tests/test_gateway.py::test_non_refusal_response_is_transparent PASSED   [ 10%]
tests/test_gateway.py::test_stream_is_forwarded_byte_for_byte_and_audited_afterward PASSED [ 13%]
tests/test_gateway.py::test_upstream_failure_is_not_retried_and_is_counted PASSED [ 17%]
tests/test_gateway.py::test_models_endpoint_forwards_to_single_upstream PASSED [ 20%]
tests/test_gateway.py::test_layered_detection_signals PASSED             [ 24%]
tests/test_gateway.py::test_verdict_cache_key_is_provider_and_model_scoped PASSED [ 27%]
tests/test_gateway.py::test_fixture_has_40_safety_framed_prompts_and_is_not_sent_upstream PASSED [ 31%]
tests/test_ladder.py::test_each_ladder_step_fires_in_order[1] PASSED     [ 34%]
tests/test_ladder.py::test_each_ladder_step_fires_in_order[2] PASSED     [ 37%]
tests/test_ladder.py::test_each_ladder_step_fires_in_order[3] PASSED     [ 41%]
tests/test_ladder.py::test_each_ladder_step_fires_in_order[4] PASSED     [ 44%]
tests/test_ladder.py::test_each_ladder_step_fires_in_order[5] PASSED     [ 48%]
tests/test_ladder.py::test_each_ladder_step_fires_in_order[6] PASSED     [ 51%]
tests/test_ladder.py::test_each_ladder_step_fires_in_order[7] PASSED     [ 55%]
tests/test_ladder.py::test_each_ladder_step_fires_in_order[8] PASSED     [ 58%]
tests/test_ladder.py::test_routes_to_next_provider_by_ascending_risk PASSED [ 62%]
tests/test_ladder.py::test_streaming_refusal_buffer_is_discarded_before_retry PASSED [ 65%]
tests/test_ladder.py::test_max_attempts_stops_after_cap PASSED           [ 68%]
tests/test_ladder.py::test_max_cost_stops_after_cost_breach PASSED       [ 72%]
tests/test_ladder.py::test_deadline_stops_loop_and_returns_best_decompose_answer PASSED [ 75%]
tests/test_ladder.py::test_all_refused_returns_full_attempt_log PASSED   [ 79%]
tests/test_ladder.py::test_observe_only_env_forwards_refusal_without_retry PASSED [ 82%]
tests/test_ladder.py::test_missing_provider_keys_are_skipped_without_crash PASSED [ 86%]
tests/test_ladder.py::test_provider_without_key_is_skipped_and_next_provider_runs PASSED [ 89%]
tests/test_ladder.py::test_unset_observe_flag_defaults_to_ladder PASSED  [ 93%]
tests/test_ladder.py::test_observe_only_loads_registry_default_upstream PASSED [ 96%]
tests/test_ladder.py::test_missing_provider_keys_are_skipped_without_crashing PASSED [100%]

============================== 29 passed in 0.93s ==============================
```
