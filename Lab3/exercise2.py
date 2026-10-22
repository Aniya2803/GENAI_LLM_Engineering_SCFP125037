
def build_review_prompt(review_text):
    prompt = f"""
You are a customer review classification assistant.

Read the customer review and assign exactly ONE category.

Categories:
- Positive: The customer expresses satisfaction or praise.
- Negative: The customer expresses dissatisfaction or complaints.
- Suggestion: The customer proposes an improvement or new feature.
- Neutral: The review is mainly factual and expresses no
  clear satisfaction, dissatisfaction, or suggestion.

Customer Review:
{review_text}

Rules:
1. Select the category that best represents the review.
2. If the review mainly proposes an improvement, use Suggestion.
3. Return only one category name.
4. Do not provide explanations or additional text.

Category:
"""
    return prompt.strip()


if __name__ == "__main__":
    review_text = (
        "The application is easy to use, but I wish it had "
        "a dark mode."
    )

    review_prompt = build_review_prompt(review_text)

    print("REVIEW CLASSIFIER PROMPT")
    print("=" * 30)
    print(review_prompt)
