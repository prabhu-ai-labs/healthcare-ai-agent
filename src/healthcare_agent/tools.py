"""Tool definitions exposed to Claude and their dispatch to the clinic store."""

import json
from collections.abc import Callable
from typing import Any

from .clinic import ClinicError, ClinicStore

DEPARTMENTS = ["primary_care", "cardiology", "dermatology"]

TOOLS: list[dict[str, Any]] = [
    {
        "name": "search_patients",
        "description": (
            "Find patients whose name contains the given text. Returns id, name and date of "
            "birth for each match. Use the date of birth to confirm identity when several match."
        ),
        "strict": True,
        "input_schema": {
            "type": "object",
            "properties": {"name": {"type": "string", "description": "Full or partial name."}},
            "required": ["name"],
            "additionalProperties": False,
        },
    },
    {
        "name": "get_patient",
        "description": "Get the full record (contact, insurance, allergies) for one patient id.",
        "strict": True,
        "input_schema": {
            "type": "object",
            "properties": {"patient_id": {"type": "string", "description": "e.g. P-1001"}},
            "required": ["patient_id"],
            "additionalProperties": False,
        },
    },
    {
        "name": "list_open_slots",
        "description": (
            "List unbooked appointment slots for a department, optionally on one date."
        ),
        "strict": True,
        "input_schema": {
            "type": "object",
            "properties": {
                "department": {"type": "string", "enum": DEPARTMENTS},
                "date": {
                    "type": ["string", "null"],
                    "description": "YYYY-MM-DD, or null for any date.",
                },
            },
            "required": ["department", "date"],
            "additionalProperties": False,
        },
    },
    {
        "name": "book_appointment",
        "description": (
            "Book an open slot for a patient. Only call this after the user has confirmed "
            "the patient, the slot and the reason for the visit."
        ),
        "strict": True,
        "input_schema": {
            "type": "object",
            "properties": {
                "patient_id": {"type": "string"},
                "slot_id": {"type": "string"},
                "reason": {"type": "string", "description": "Short reason for the visit."},
            },
            "required": ["patient_id", "slot_id", "reason"],
            "additionalProperties": False,
        },
    },
]


def run_tool(store: ClinicStore, name: str, tool_input: dict[str, Any]) -> tuple[str, bool]:
    """Execute a tool call. Returns ``(content, is_error)`` for the tool_result block."""
    handlers: dict[str, Callable[..., Any]] = {
        "search_patients": store.search_patients,
        "get_patient": store.get_patient,
        "list_open_slots": store.list_open_slots,
        "book_appointment": store.book_appointment,
    }
    handler = handlers.get(name)
    if handler is None:
        return f"Unknown tool {name!r}.", True
    try:
        return json.dumps(handler(**tool_input)), False
    except (ClinicError, TypeError) as exc:
        return f"Error: {exc}", True
