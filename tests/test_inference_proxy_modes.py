import inference
import pytest


def test_make_openai_client_rejects_placeholder_token(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("API_BASE_URL", "https://router.huggingface.co/v1")
    monkeypatch.setenv("API_KEY", "hf_your_actual_token_here")
    monkeypatch.delenv("HF_TOKEN", raising=False)

    client, _, startup_error = inference._make_openai_client()

    assert client is None
    assert startup_error is not None
    assert "placeholder token" in startup_error.lower()


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
