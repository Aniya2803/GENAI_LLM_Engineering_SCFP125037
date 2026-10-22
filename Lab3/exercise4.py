
def build_support_prompt(customer_message, company_name, max_words):
    prompt = f"""
You are a professional customer support representative
for {company_name}.

Customer message:
{customer_message}

Write a helpful reply following these rules:
1. Use a polite, empathetic, professional tone.
2. Acknowledge the customer's concern.
3. Provide a relevant response based only on the information given.
4. Do not invent policies, refunds, or promises.
5. Keep the reply within {max_words} words.
6. Return only the reply, without a heading or explanation.

Support reply:
"""
    return prompt.strip()


if __name__ == "__main__":
    customer_message = (
        "My internet connection has stopped working "
        "since yesterday. Please help."
    )

    company_name = "ConnectPlus"
    max_words = 60

    support_prompt = build_support_prompt(
        customer_message,
        company_name,
        max_words
    )

    print("CUSTOMER SUPPORT PROMPT")
    print("=" * 30)
    print(support_prompt)
