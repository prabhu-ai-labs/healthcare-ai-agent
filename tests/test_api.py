import pytest
from conftest import message, text_block
from fastapi.testclient import TestClient

from healthcare_agent.api import app, get_client
from healthcare_agent.config import Settings, get_settings
from healthcare_agent.extraction import Insurance, IntakeExtraction


@pytest.fixture
def api(fake_client, monkeypatch):
    monkeypatch.setattr("healthcare_agent.api.build_client", lambda: fake_client)
    app.dependency_overrides[get_client] = lambda: fake_client
    app.dependency_overrides[get_settings] = lambda: Settings(_env_file=None)
    with TestClient(app) as client:
        yield client
    app.dependency_overrides.clear()


def test_health(api):
    assert api.get("/health").json()["status"] == "ok"


def test_chat_keeps_conversation(api, fake_client):
    fake_client.messages.responses = [message([text_block("Hi!")]), message([text_block("Sure.")])]
    first = api.post("/chat", json={"message": "hello"}).json()
    second = api.post(
        "/chat", json={"message": "book me", "conversation_id": first["conversation_id"]}
    ).json()

    assert first["reply"] == "Hi!"
    assert second["conversation_id"] == first["conversation_id"]
    assert len(fake_client.messages.calls[1]["messages"]) == 3


def test_chat_refusal_maps_to_422(api, fake_client):
    fake_client.messages.responses = [message([], stop_reason="refusal")]
    assert api.post("/chat", json={"message": "x"}).status_code == 422


def test_extract_intake_text(api, fake_client):
    parsed = IntakeExtraction(
        patient_name="Maria Alvarez",
        date_of_birth="1984-03-12",
        phone="555-0142",
        insurance=Insurance(provider="Acme Health", member_id="AH-882134"),
        reason_for_visit="Persistent cough",
        current_medications=[],
        allergies=["penicillin"],
        urgency="routine",
        missing_fields=[],
    )
    fake_client.messages.responses = [message([], parsed_output=parsed)]

    response = api.post(
        "/extract/intake", files={"file": ("intake.txt", b"Name: Maria Alvarez", "text/plain")}
    )

    assert response.status_code == 200
    assert response.json()["allergies"] == ["penicillin"]
    call = fake_client.messages.calls[0]
    assert call["output_format"] is IntakeExtraction
    assert call["messages"][0]["content"][0]["source"]["data"] == "Name: Maria Alvarez"


def test_extract_rejects_unsupported_type(api):
    response = api.post("/extract/intake", files={"file": ("x.png", b"\x89PNG", "image/png")})
    assert response.status_code == 415
