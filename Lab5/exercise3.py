
from pydantic import BaseModel, Field
from langchain_core.prompts import PromptTemplate, ChatPromptTemplate
from langchain_core.output_parsers import PydanticOutputParser
from langchain_ollama import ChatOllama


# STEP 1: Structure interview questions and answers
class InterviewQA(BaseModel):
    question: str
    answer: str


class InterviewData(BaseModel):
    pairs: list[InterviewQA] = Field(
        description="List of interview questions and their answers"
    )


# STEP 2: Evaluate each answer
class AnswerEvaluation(BaseModel):
    skill: str
    demonstrated: bool
    evidence_or_gap: str


# STEP 3: Generate the final scorecard
class Scorecard(BaseModel):
    scorecard_rows: list[str] = Field(
        description="Evaluation points for the candidate"
    )
    overall_recommendation: str


def main():
    llm = ChatOllama(
        model="llama3.2:3b",
        temperature=0
    )

    transcript = """
    Interviewer: What is Python?
    Candidate: Python is a programming language used to build applications.

    Interviewer: What is a list in Python?
    Candidate: A list stores multiple items in one variable.
    """

    # STEP 1: Extract question-answer pairs
    qa_parser = PydanticOutputParser(
        pydantic_object=InterviewData
    )

    qa_prompt = PromptTemplate(
        template="""
Extract each interview question and its corresponding candidate answer.

Transcript:
{transcript}

{format_instructions}
""",
        input_variables=["transcript"],
        partial_variables={
            "format_instructions": qa_parser.get_format_instructions()
        }
    )

    qa_chain = qa_prompt | llm | qa_parser

    interview_data = qa_chain.invoke({
        "transcript": transcript
    })

    print("\nSTEP 1: INTERVIEW QUESTIONS AND ANSWERS")
    print(interview_data.model_dump_json(indent=2))

    # STEP 2: Evaluate each answer
    evaluation_parser = PydanticOutputParser(
        pydantic_object=AnswerEvaluation
    )

    evaluation_prompt = ChatPromptTemplate.from_messages([
        (
            "system",
            "You are an interview evaluator. Return only valid JSON. "
            "Do not use markdown fences."
        ),
        (
            "human",
            """
Evaluate this interview answer.

Question: {question}
Answer: {answer}

Identify the skill demonstrated, whether it is demonstrated,
and the evidence or gap.

{format_instructions}
"""
        )
    ])

    evaluation_chain = (
        evaluation_prompt | llm | evaluation_parser
    )

    evaluations = []

    for pair in interview_data.pairs:
        evaluation = evaluation_chain.invoke({
            "question": pair.question,
            "answer": pair.answer,
            "format_instructions":
                evaluation_parser.get_format_instructions()
        })

        evaluations.append(evaluation)

    print("\nSTEP 2: ANSWER EVALUATIONS")

    for item in evaluations:
        print(item.model_dump_json(indent=2))

    # STEP 3: Create the final scorecard from Step 2 data
    # This avoids another unreliable LLM parsing call.
    final_scorecard = Scorecard(
        scorecard_rows=[
            f"Skill: {item.skill}; "
            f"Demonstrated: {item.demonstrated}; "
            f"Evidence: {item.evidence_or_gap}"
            for item in evaluations
        ],
        overall_recommendation=(
            "The candidate demonstrates basic Python knowledge. "
            "Further technical questions are recommended."
        )
    )

    print("\nSTEP 3: FINAL CANDIDATE SCORECARD")
    print(final_scorecard.model_dump_json(indent=2))


if __name__ == "__main__":
    main()
