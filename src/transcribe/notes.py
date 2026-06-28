from __future__ import annotations

from openai import OpenAI

from .config import Settings


def generate_summary(settings: Settings, transcript_markdown: str, title: str) -> str:
    provider = settings.summary_provider.strip().lower()
    if provider == "none":
        raise RuntimeError("Summary generation is disabled.")
    if provider != "openai":
        raise RuntimeError(f"Unsupported summary provider: {settings.summary_provider}")
    if not settings.openai_api_key:
        raise RuntimeError("OPENAI_API_KEY is required when SUMMARY_PROVIDER=openai")

    client = OpenAI(api_key=settings.openai_api_key)
    response = client.responses.create(
        model=settings.openai_summary_model,
        input=[
            {
                "role": "system",
                "content": [
                    {
                        "type": "input_text",
                        "text": (
                            "You generate concise but useful meeting notes from transcripts. "
                            "Return markdown with sections: Summary, Decisions, Action Items, Risks / Blockers, Open Questions. "
                            "If the transcript is noisy, say so instead of inventing specifics."
                        ),
                    }
                ],
            },
            {
                "role": "user",
                "content": [
                    {
                        "type": "input_text",
                        "text": f"Meeting title: {title}\n\nTranscript:\n\n{transcript_markdown}",
                    }
                ],
            },
        ],
    )
    return response.output_text.strip() + "\n"
