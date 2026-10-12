"""Document intelligence: turn a patient intake document into validated structured data."""

import base64
from typing import Any, Literal

import anthropic
from pydantic import BaseModel, Field

from .config import Settings
from .llm import raise_if_refused, request_options

EXTRACTION_PROMPT = """\
Extract the patient intake information from the attached document into the requested schema.

- Copy values as written; do not infer or invent anything that is not in the document.
- Use null for any single field the document does not contain, and list each such field \
name in missing_fields.
- Dates use YYYY-MM-DD.
- urgency: "emergency" only if the document describes symptoms that need emergency care \
now, "urgent" if it asks to be seen within days, otherwise "routine"."""


class UnsupportedDocumentError(ValueError):
    """The uploaded file type cannot be sent to Claude as a document."""


class Insurance(BaseModel):
    provider: str | None
    member_id: str | None


class Medication(BaseModel):
    name: str
    dose: str | None = Field(description="Dose and frequency as written, e.g. '10 mg daily'.")


class IntakeExtraction(BaseModel):
    patient_name: str | None
    date_of_birth: str | None = Field(description="YYYY-MM-DD")
    phone: str | None
    insurance: Insurance
    reason_for_visit: str | None
    current_medications: list[Medication]
    allergies: list[str]
    urgency: Literal["routine", "urgent", "emergency"]
    missing_fields: list[str]


def _document_block(content: bytes, media_type: str) -> dict[str, Any]:
    if media_type == "application/pdf":
        return {
            "type": "document",
            "source": {
                "type": "base64",
                "media_type": "application/pdf",
                "data": base64.standard_b64encode(content).decode("ascii"),
            },
        }
    if media_type.startswith("text/"):
        return {
            "type": "document",
            "source": {
                "type": "text",
                "media_type": "text/plain",
                "data": content.decode("utf-8", errors="replace"),
            },
        }
    raise UnsupportedDocumentError(
        f"Unsupported document type {media_type!r}; use PDF or plain text."
    )


def extract_intake(
    client: anthropic.Anthropic, settings: Settings, content: bytes, media_type: str
) -> IntakeExtraction:
    response = client.beta.messages.parse(
        **request_options(settings, settings.extraction_effort),
        messages=[
            {
                "role": "user",
                "content": [
                    _document_block(content, media_type),
                    {"type": "text", "text": EXTRACTION_PROMPT},
                ],
            }
        ],
        output_format=IntakeExtraction,
    )
    raise_if_refused(response)
    if response.parsed_output is None:
        raise ValueError(f"No structured output returned (stop_reason={response.stop_reason}).")
    return response.parsed_output
