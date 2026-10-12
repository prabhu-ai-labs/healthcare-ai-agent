"""In-memory clinic data store backed by synthetic records.

Stands in for an EHR / practice-management system so the agent can be exercised end to end
without touching real patient data.
"""

import json
import threading
from dataclasses import dataclass, field
from importlib import resources
from typing import Any


class ClinicError(ValueError):
    """A request against the clinic store that cannot be fulfilled."""


@dataclass
class ClinicStore:
    patients: dict[str, dict[str, Any]]
    slots: dict[str, dict[str, Any]]
    appointments: dict[str, dict[str, Any]] = field(default_factory=dict)
    _lock: threading.Lock = field(default_factory=threading.Lock, repr=False)

    @classmethod
    def from_synthetic_data(cls) -> "ClinicStore":
        raw = resources.files("healthcare_agent").joinpath("data/synthetic_clinic.json").read_text()
        data = json.loads(raw)
        return cls(
            patients={p["id"]: p for p in data["patients"]},
            slots={s["id"]: {**s, "booked": False} for s in data["slots"]},
        )

    def search_patients(self, name: str) -> list[dict[str, Any]]:
        needle = name.strip().lower()
        if not needle:
            raise ClinicError("A name to search for is required.")
        return [
            {"id": p["id"], "name": p["name"], "date_of_birth": p["date_of_birth"]}
            for p in self.patients.values()
            if needle in p["name"].lower()
        ]

    def get_patient(self, patient_id: str) -> dict[str, Any]:
        try:
            return self.patients[patient_id]
        except KeyError:
            raise ClinicError(f"No patient with id {patient_id!r}.") from None

    def list_open_slots(self, department: str, date: str | None = None) -> list[dict[str, Any]]:
        return [
            {k: v for k, v in s.items() if k != "booked"}
            for s in self.slots.values()
            if s["department"] == department
            and not s["booked"]
            and (date is None or s["start"].startswith(date))
        ]

    def book_appointment(self, patient_id: str, slot_id: str, reason: str) -> dict[str, Any]:
        self.get_patient(patient_id)
        with self._lock:
            slot = self.slots.get(slot_id)
            if slot is None:
                raise ClinicError(f"No slot with id {slot_id!r}.")
            if slot["booked"]:
                raise ClinicError(f"Slot {slot_id} is already booked.")
            slot["booked"] = True
            appointment_id = f"A-{len(self.appointments) + 1:04d}"
            appointment = {
                "id": appointment_id,
                "patient_id": patient_id,
                "slot_id": slot_id,
                "department": slot["department"],
                "provider": slot["provider"],
                "start": slot["start"],
                "reason": reason,
            }
            self.appointments[appointment_id] = appointment
            return appointment
