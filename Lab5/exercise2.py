
from pydantic import BaseModel, Field
from langchain_core.prompts import PromptTemplate, ChatPromptTemplate
from langchain_core.output_parsers import PydanticOutputParser, StrOutputParser
from langchain_ollama import ChatOllama


# Step 1: Define the structure for research information
class ResearchData(BaseModel):
    research_question: str = Field(
        description="Main question the research investigates"
    )
    method: str = Field(
        description="Method used in the research"
    )
    key_finding: str = Field(
        description="Most important result of the research"
    )


def main():
    llm = ChatOllama(
        model="llama3.2:3b",
        temperature=0
    )

    abstract = """
    This study investigates whether regular physical activity
    improves sleep quality among university students. Researchers
    collected survey responses from 200 students and compared
    their activity levels with reported sleep quality. The results
    showed that students reporting regular physical activity also
    reported better sleep quality.
    """

    # STEP 1: Extract structured research information
    parser = PydanticOutputParser(
        pydantic_object=ResearchData
    )

    extraction_prompt = PromptTemplate(
        template="""
Extract the research question, method, and key finding
from the abstract below.

Abstract:
{abstract}

{format_instructions}
""",
        input_variables=["abstract"],
        partial_variables={
            "format_instructions": parser.get_format_instructions()
        }
    )

    extraction_chain = extraction_prompt | llm | parser

    research_data = extraction_chain.invoke({
        "abstract": abstract
    })

    print("\nSTEP 1: STRUCTURED RESEARCH DATA")
    print(research_data.model_dump_json(indent=2))

    # STEP 2: Explain the findings in simple language
    summary_prompt = ChatPromptTemplate.from_messages([
        (
            "system",
            "You are a science communicator explaining research "
            "to a general audience. Use simple, clear language."
        ),
        (
            "human",
            """
Research question: {research_question}
Method: {method}
Key finding: {key_finding}

Explain this research in simple language for a non-expert.
Do not add information that is not provided.
"""
        )
    ])

    summary_chain = summary_prompt | llm | StrOutputParser()

    summary = summary_chain.invoke(
        research_data.model_dump()
    )

    print("\nSTEP 2: LAYPERSON SUMMARY")
    print(summary)


if __name__ == "__main__":
    main()
