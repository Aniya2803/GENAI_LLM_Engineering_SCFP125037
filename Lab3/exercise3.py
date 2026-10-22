
import json


def build_resume_prompt(resume_text, fields_to_extract):
    fields = ", ".join(fields_to_extract)

    prompt = f"""
You are a resume information extraction assistant.

Extract the requested fields from the resume below.

Resume:
{resume_text}

Fields to extract:
{fields}

Instructions:
1. Return a valid JSON object only.
2. Include only the requested fields as JSON keys.
3. Use the exact field names provided.
4. If a field is missing, use null as its value.
5. Do not guess or invent information.
6. Do not add Markdown fences or explanations.
"""
    return prompt.strip()


if __name__ == "__main__":
    resume_text = """
    Name: Rahul Sharma
    Email: rahul@example.com
    Skills: Python, SQL, Machine Learning
    Education: MCA
    """

    fields_to_extract = ["Name", "Email", "Skills"]

    resume_prompt = build_resume_prompt(
        resume_text,
        fields_to_extract
    )

    print("RESUME FIELD EXTRACTOR PROMPT")
    print("=" * 35)
    print(resume_prompt)
