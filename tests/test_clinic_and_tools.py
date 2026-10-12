import json

import pytest

from healthcare_agent.clinic import ClinicError
from healthcare_agent.tools import TOOLS, run_tool


def test_search_patients_matches_partial_name(clinic):
    names = {p["name"] for p in clinic.search_patients("maria")}
    assert names == {"Maria Alvarez", "Maria Chen"}


def test_list_open_slots_filters_by_date(clinic):
    slots = clinic.list_open_slots("primary_care", "2026-10-14")
    assert [s["id"] for s in slots] == ["S-01", "S-02"]


def test_booking_marks_slot_taken(clinic):
    appointment = clinic.book_appointment("P-1001", "S-01", "Annual physical")
    assert appointment["provider"] == "Dr. Lena Hart"
    assert "S-01" not in {s["id"] for s in clinic.list_open_slots("primary_care", None)}
    with pytest.raises(ClinicError, match="already booked"):
        clinic.book_appointment("P-1004", "S-01", "Follow-up")


def test_booking_unknown_patient_fails(clinic):
    with pytest.raises(ClinicError, match="No patient"):
        clinic.book_appointment("P-9999", "S-02", "Checkup")


def test_run_tool_returns_json(clinic):
    content, is_error = run_tool(clinic, "get_patient", {"patient_id": "P-1002"})
    assert not is_error
    assert json.loads(content)["name"] == "James Okafor"


def test_run_tool_reports_errors(clinic):
    assert run_tool(clinic, "get_patient", {"patient_id": "nope"})[1] is True
    assert run_tool(clinic, "no_such_tool", {})[1] is True
    assert run_tool(clinic, "get_patient", {"wrong_arg": "x"})[1] is True


def test_tool_schemas_are_strict():
    for tool in TOOLS:
        schema = tool["input_schema"]
        assert tool["strict"] is True
        assert schema["additionalProperties"] is False
        assert set(schema["required"]) == set(schema["properties"])
