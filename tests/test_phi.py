from healthcare_agent.phi import redact


def test_redacts_direct_identifiers():
    text = "SSN 123-45-6789, call 555-014-2233, mail a@b.org, DOB 1984-03-12, MRN: 88231"
    assert redact(text) == "SSN [SSN], call [PHONE], mail [EMAIL], DOB [DATE], [MRN]"


def test_leaves_plain_text_alone():
    assert redact("Patient reports a mild cough.") == "Patient reports a mild cough."
