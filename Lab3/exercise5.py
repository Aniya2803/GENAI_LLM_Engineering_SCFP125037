
def build_translation_prompt(sentence, target_language, formality):
    prompt = f"""
You are a multilingual translation assistant.

Translate the following sentence into {target_language}.

Original sentence:
{sentence}

Required formality: {formality}

Instructions:
1. Preserve the original meaning.
2. Match the requested formality level.
3. Use natural and grammatically correct language.
4. Do not add explanations or extra information.
5. Return only the translated sentence.

Translation:
"""
    return prompt.strip()


if __name__ == "__main__":
    sentence = "Could you please help me with this task?"
    target_language = "Hindi"
    formality = "Formal"

    translation_prompt = build_translation_prompt(
        sentence,
        target_language,
        formality
    )

    print("TRANSLATION PROMPT")
    print("=" * 30)
    print(translation_prompt)
