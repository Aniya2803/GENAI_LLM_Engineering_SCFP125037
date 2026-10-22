
def refine_draft(draft_text, issue_to_fix, round_number):
    prompt = f"""
You are a writing improvement assistant.

Current draft:
{draft_text}

Issue to fix:
{issue_to_fix}

Round: {round_number}

Improve only the specified issue. Preserve the original
meaning and return only the revised draft.
"""
    return prompt.strip()


draft_text = (
    "Our application is good. It helps users do work. "
    "Many people can use it."
)

issue_to_fix = "Improve clarity and remove vague wording."

# Demonstration of successive revisions
revisions = [
    "Our application is useful. It helps users complete tasks.",
    "Our application helps users complete tasks efficiently.",
    "Our application enables users to complete their tasks more clearly and efficiently."
]

for round_number in range(1, 4):
    prompt = refine_draft(
        draft_text,
        issue_to_fix,
        round_number
    )

    print(f"\nREFINEMENT ROUND {round_number}")
    print("=" * 30)
    print(prompt)

    # Demonstration only: this revision is predefined.
    draft_text = revisions[round_number - 1]
    print("\nRevised draft:")
    print(draft_text)
