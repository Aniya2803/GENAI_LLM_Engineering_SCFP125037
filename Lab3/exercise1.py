
def build_doctor_prompt(patient_name, patient_notes):
    prompt = f"""
You are an assistant that organizes patient consultation notes.

Patient Name: {patient_name}

Patient Notes:
{patient_notes}

Create a concise doctor's summary using exactly
the following three sections:

Symptoms:
Diagnosis:
Recommendation:

Rules:
1. Include only information supported by the notes.
2. Do not invent symptoms or diagnoses.
3. If information is missing, write "Not provided".
4. Keep the language clear and professional.
5. Do not add extra headings or explanations.

Return only the formatted summary.
"""
    return prompt.strip()


# Sample input
patient_name = "Rahul Sharma"

patient_notes = """
The patient reports a cough and mild fever for two days.
A diagnosis has not been provided.
The notes recommend consulting a qualified doctor.
"""

# Build and display the prompt
doctor_prompt = build_doctor_prompt(
    patient_name,
    patient_notes
)

print("DOCTOR SUMMARY PROMPT")
print("=" * 30)
print(doctor_prompt)
