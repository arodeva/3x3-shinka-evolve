"""Regression tests for Anthropic pricing parity and response block parsing."""

import asyncio
from types import SimpleNamespace

import anthropic
import httpx as httpx2  # anthropic's SDK types its signatures against this alias
import pytest

import shinka.llm.providers.anthropic as anthropic_provider
from shinka.llm.providers.anthropic import (
    get_anthropic_costs,
    query_anthropic,
    query_anthropic_async,
)


def _adaptive_thinking_rejection() -> anthropic.BadRequestError:
    """The 400 a newer Claude model (e.g. one requiring adaptive thinking)
    returns when sent an explicit thinking budget -- see
    _wants_adaptive_thinking / _adapt_thinking_kwargs in the provider."""
    response = httpx2.Response(
        400, request=httpx2.Request("POST", "https://api.anthropic.com/v1/messages")
    )
    message = (
        'Error code: 400 - {\'error\': {\'message\': \'"thinking.type.enabled" '
        'is not supported for this model. Use "thinking.type.adaptive" and '
        '"output_config.effort" to control thinking behavior.\'}}'
    )
    return anthropic.BadRequestError(message, response=response, body=None)


class _FlakyAdaptiveClient:
    """Rejects an explicit thinking budget once, then succeeds -- records the
    kwargs passed to each attempt so the retry's correction can be checked."""

    def __init__(self, response, *, asynchronous: bool = False):
        self.calls: list[dict] = []

        def record_and_respond(**kwargs):
            self.calls.append(kwargs)
            if len(self.calls) == 1:
                raise _adaptive_thinking_rejection()
            return response

        async def create_async(**kwargs):
            return record_and_respond(**kwargs)

        create = create_async if asynchronous else record_and_respond
        self.messages = SimpleNamespace(create=create)


def _response(
    input_tokens: int = 10,
    output_tokens: int = 20,
    thinking_tokens: int | None = None,
):
    details = (
        None
        if thinking_tokens is None
        else SimpleNamespace(thinking_tokens=thinking_tokens)
    )
    return SimpleNamespace(
        content=[SimpleNamespace(type="text", text="ok")],
        usage=SimpleNamespace(
            input_tokens=input_tokens,
            output_tokens=output_tokens,
            output_tokens_details=details,
        ),
    )


class _Client:
    def __init__(self, response, *, asynchronous: bool = False):
        async def create_async(**_kwargs):
            return response

        def create_sync(**_kwargs):
            return response

        create = create_async if asynchronous else create_sync
        self.messages = SimpleNamespace(create=create)


def test_unknown_model_preserves_tokens_and_defaults_cost_to_zero():
    costs = get_anthropic_costs(_response(), "anthropic/not-in-catalog")

    assert costs == {
        "input_tokens": 10,
        "output_tokens": 20,
        "thinking_tokens": 0,
        "input_cost": 0.0,
        "output_cost": 0.0,
        "cost": 0.0,
    }


def test_thinking_tokens_are_separated_but_billed_as_output(monkeypatch):
    billed_usage = []

    def calculate_cost(model, input_tokens, output_tokens):
        billed_usage.append((model, input_tokens, output_tokens))
        return 1.0, 2.0

    monkeypatch.setattr(anthropic_provider, "model_exists", lambda _model: True)
    monkeypatch.setattr(anthropic_provider, "calculate_cost", calculate_cost)

    costs = get_anthropic_costs(_response(thinking_tokens=7), "claude-test")

    assert costs == {
        "input_tokens": 10,
        "output_tokens": 13,
        "thinking_tokens": 7,
        "input_cost": 1.0,
        "output_cost": 2.0,
        "cost": 3.0,
    }
    assert billed_usage == [("claude-test", 10, 20)]


def test_thinking_tokens_are_bounded_by_billed_output():
    costs = get_anthropic_costs(
        _response(output_tokens=20, thinking_tokens=25),
        "anthropic/not-in-catalog",
    )

    assert costs["output_tokens"] == 0
    assert costs["thinking_tokens"] == 20


def test_sync_and_async_share_thinking_accounting():
    response = _response(thinking_tokens=7)
    sync_result = query_anthropic(
        _Client(response), "anthropic/not-in-catalog", "msg", "sys", [], None
    )
    async_result = asyncio.run(
        query_anthropic_async(
            _Client(response, asynchronous=True),
            "anthropic/not-in-catalog",
            "msg",
            "sys",
            [],
            None,
        )
    )

    assert sync_result.output_tokens == async_result.output_tokens == 13
    assert sync_result.thinking_tokens == async_result.thinking_tokens == 7


def _blocks_response(blocks):
    return SimpleNamespace(
        content=[SimpleNamespace(type=name, **attrs) for name, attrs in blocks],
        usage=SimpleNamespace(
            input_tokens=10,
            output_tokens=20,
            output_tokens_details=None,
        ),
    )


CONTENT_BLOCK_CASES = [
    pytest.param([("text", {"text": "a"})], "a", "", id="text-only"),
    pytest.param(
        [("thinking", {"thinking": "t"}), ("text", {"text": "a"})],
        "a",
        "t",
        id="thinking-then-text",
    ),
    pytest.param(
        [
            ("thinking", {"thinking": "t"}),
            ("redacted_thinking", {"data": "e"}),
            ("text", {"text": "a"}),
        ],
        "a",
        "t\ne",
        id="redacted-thinking-in-the-middle",
    ),
    pytest.param(
        [("redacted_thinking", {"data": "e"}), ("text", {"text": "a"})],
        "a",
        "e",
        id="redacted-thinking-first",
    ),
    pytest.param([("thinking", {"thinking": "t"})], "", "t", id="thinking-only"),
    pytest.param(
        [
            ("thinking", {"thinking": "t"}),
            ("text", {"text": "p1"}),
            ("text", {"text": "p2"}),
        ],
        "p1\np2",
        "t",
        id="several-text-blocks",
    ),
    pytest.param(
        [("text", {"text": "a"}), ("tool_use", {"id": "u1", "name": "f", "input": {}})],
        "a",
        "",
        id="unhandled-block-type-is-skipped",
    ),
]


@pytest.mark.parametrize(
    "blocks, expected_content, expected_thought", CONTENT_BLOCK_CASES
)
def test_content_blocks_are_split_by_type(blocks, expected_content, expected_thought):
    result = query_anthropic(
        _Client(_blocks_response(blocks)),
        "anthropic/not-in-catalog",
        "msg",
        "sys",
        [],
        None,
    )

    assert result.content == expected_content
    assert result.thought == expected_thought


@pytest.mark.parametrize(
    "blocks, expected_content, expected_thought", CONTENT_BLOCK_CASES
)
def test_async_content_blocks_are_split_by_type(
    blocks, expected_content, expected_thought
):
    result = asyncio.run(
        query_anthropic_async(
            _Client(_blocks_response(blocks), asynchronous=True),
            "anthropic/not-in-catalog",
            "msg",
            "sys",
            [],
            None,
        )
    )

    assert result.content == expected_content
    assert result.thought == expected_thought


def test_adaptive_thinking_rejection_retries_with_corrected_kwargs():
    client = _FlakyAdaptiveClient(_response())
    result = query_anthropic(
        client,
        "anthropic/not-in-catalog",
        "msg",
        "sys",
        [],
        None,
        thinking={"type": "enabled", "budget_tokens": 4096},
        max_tokens=32768,
    )

    assert result.content == "ok"
    assert len(client.calls) == 2
    assert client.calls[0]["thinking"] == {"type": "enabled", "budget_tokens": 4096}
    assert client.calls[1]["thinking"] == {"type": "adaptive"}
    # Unrelated kwargs must survive the retry unchanged.
    assert client.calls[1]["max_tokens"] == 32768


def test_async_adaptive_thinking_rejection_retries_with_corrected_kwargs():
    client = _FlakyAdaptiveClient(_response(), asynchronous=True)
    result = asyncio.run(
        query_anthropic_async(
            client,
            "anthropic/not-in-catalog",
            "msg",
            "sys",
            [],
            None,
            thinking={"type": "enabled", "budget_tokens": 8192},
            max_tokens=16384,
        )
    )

    assert result.content == "ok"
    assert len(client.calls) == 2
    assert client.calls[0]["thinking"] == {"type": "enabled", "budget_tokens": 8192}
    assert client.calls[1]["thinking"] == {"type": "adaptive"}
    assert client.calls[1]["max_tokens"] == 16384


def test_unrelated_bad_request_error_is_not_swallowed(monkeypatch):
    """A 400 for a different reason must propagate, not be mistaken for the
    adaptive-thinking case and silently retried with altered kwargs.

    BadRequestError is an APIStatusError, which the module's @backoff.on_exception
    retries up to MAX_TRIES times -- pointlessly, since a 400 is deterministic,
    but that's existing, unrelated behavior this test isn't about. Patch
    time.sleep so backoff's real exponential waits don't make this test slow.
    """
    monkeypatch.setattr("time.sleep", lambda _seconds: None)
    response = httpx2.Response(
        400, request=httpx2.Request("POST", "https://api.anthropic.com/v1/messages")
    )
    other_error = anthropic.BadRequestError(
        "Error code: 400 - some unrelated validation error", response=response, body=None
    )

    def always_fails(**_kwargs):
        raise other_error

    client = SimpleNamespace(messages=SimpleNamespace(create=always_fails))
    with pytest.raises(anthropic.BadRequestError):
        query_anthropic(
            client,
            "anthropic/not-in-catalog",
            "msg",
            "sys",
            [],
            None,
            thinking={"type": "enabled", "budget_tokens": 4096},
        )
