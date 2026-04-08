import inference
import pytest


def test_make_openai_client_rejects_placeholder_token(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("API_BASE_URL", "https://router.huggingface.co/v1")
    monkeypatch.setenv("HF_TOKEN", "hf_your_actual_token_here")

    client, _, startup_error = inference._make_openai_client()

    assert client is None
    assert startup_error is not None
    assert "placeholder token" in startup_error.lower()


def test_env_resolution_uses_hf_token_and_defaults(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("API_BASE_URL", raising=False)
    monkeypatch.delenv("MODEL_NAME", raising=False)
    monkeypatch.delenv("HF_TOKEN", raising=False)
    monkeypatch.setenv("API_KEY", "hf_should_not_be_used")

    assert inference._resolve_api_base_url() == inference.DEFAULT_HF_ROUTER_BASE_URL
    assert inference._resolve_model_name() == inference.DEFAULT_MODEL_NAME
    assert inference._resolve_hf_token() is None

    monkeypatch.setenv("HF_TOKEN", "hf_live_token")
    assert inference._resolve_hf_token() == "hf_live_token"


def test_run_benchmark_strict_mode_raises_when_proxy_unavailable(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(
        inference,
        "_cached_openai_client",
        lambda: (None, "Qwen/Qwen2.5-72B-Instruct", "proxy init failed"),
    )

    with pytest.raises(RuntimeError, match="proxy init failed"):
        inference.run_benchmark(["easy"], require_remote_llm=True)


def test_run_benchmark_non_strict_mode_falls_back(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(
        inference,
        "_cached_openai_client",
        lambda: (None, "Qwen/Qwen2.5-72B-Instruct", "proxy init failed"),
    )
    monkeypatch.delenv("REQUIRE_REMOTE_LLM", raising=False)
    monkeypatch.delenv("STRICT_PROXY_MODE", raising=False)

    result = inference.run_benchmark(["easy"], require_remote_llm=False)

    assert result["client_mode"] == "baseline_fallback"
    assert result["startup_error"] == "proxy init failed"
    assert result["require_remote_llm"] is False


def test_run_benchmark_stdout_is_structured_only(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    monkeypatch.setattr(
        inference,
        "_cached_openai_client",
        lambda: (None, "Qwen/Qwen2.5-72B-Instruct", "proxy init failed"),
    )

    inference.run_benchmark(["easy"], require_remote_llm=False)
    output = capsys.readouterr().out

    assert "[START]" in output
    assert "[STEP]" in output
    assert "[END]" in output
    assert "[INFO]" not in output
    assert "[WARN]" not in output
    assert "[FATAL]" not in output
