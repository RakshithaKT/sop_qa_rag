
from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest

from sop_qa.generation.response import (
    AnswerGenerator,
    RetrievedEvidence,
)


@pytest.mark.asyncio
async def test_empty_evidence_does_not_call_llm():
    client = SimpleNamespace(
        chat=SimpleNamespace(
            completions=SimpleNamespace(
                create=AsyncMock()
            )
        )
    )

    generator = AnswerGenerator(
        client=client,
        model="test-model",
    )

    result = await generator.generate(
        "What is the machine startup procedure?",
        [],
    )

    assert "could not find relevant evidence" in result.answer
    assert result.sources == []

    client.chat.completions.create.assert_not_called()


@pytest.mark.asyncio
async def test_answer_uses_retrieved_evidence():
    client = SimpleNamespace(
        chat=SimpleNamespace(
            completions=SimpleNamespace(
                create=AsyncMock(
                    return_value=SimpleNamespace(
                        choices=[
                            SimpleNamespace(
                                message=SimpleNamespace(
                                    content=(
                                        "Follow the startup checklist [S1]."
                                    )
                                )
                            )
                        ]
                    )
                )
            )
        )
    )

    generator = AnswerGenerator(
        client=client,
        model="test-model",
    )

    evidence = [
        RetrievedEvidence(
            text="Complete the startup checklist before operation.",
            source="machine_sop.pdf",
            page=3,
            chunk_id="chunk-001",
        )
    ]

    result = await generator.generate(
        "What should I do before operation?",
        evidence,
    )

    assert "[S1]" in result.answer
    assert result.sources == evidence

    call = client.chat.completions.create.call_args
    assert call.kwargs["temperature"] == 0
    assert "machine_sop.pdf" in call.kwargs["messages"][1]["content"]
    assert "page: 3" in call.kwargs["messages"][1]["content"]


@pytest.mark.asyncio
async def test_empty_llm_response_raises_error():
    client = SimpleNamespace(
        chat=SimpleNamespace(
            completions=SimpleNamespace(
                create=AsyncMock(
                    return_value=SimpleNamespace(
                        choices=[
                            SimpleNamespace(
                                message=SimpleNamespace(content=" ")
                            )
                        ]
                    )
                )
            )
        )
    )

    generator = AnswerGenerator(
        client=client,
        model="test-model",
    )

    with pytest.raises(RuntimeError, match="empty answer"):
        await generator.generate(
            "Explain the procedure.",
            [
                RetrievedEvidence(
                    text="Example evidence",
                    source="example.pdf",
                    page=1,
                )
            ],
        )
