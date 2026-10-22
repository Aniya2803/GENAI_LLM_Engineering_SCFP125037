
from typing import Literal

from pydantic import BaseModel, Field
from langchain_core.prompts import PromptTemplate, ChatPromptTemplate
from langchain_core.output_parsers import PydanticOutputParser, StrOutputParser
from langchain_ollama import ChatOllama


# Define the structure for extracted customer review data
class ComplaintData(BaseModel):
    complaint: str = Field(description="Main issue reported by the customer")
    product_or_feature: str = Field(
        description="Product or feature mentioned in the review"
    )
    sentiment: Literal["Positive", "Negative", "Neutral", "Mixed"] = Field(
        description="Overall sentiment expressed by the customer"
    )


# Main workflow
def main():

    # Connect to the local Ollama model
    llm = ChatOllama(
        model="llama3.2:3b",
        temperature=0
    )

    review_text = (
        "The mobile app keeps crashing when I upload photos. "
        "I like the design, but this problem is frustrating."
    )

    # STEP 1: Extract structured information
    parser = PydanticOutputParser(
        pydantic_object=ComplaintData
    )

    extraction_prompt = PromptTemplate(
        template="""
Extract the complaint, product or feature, and customer sentiment
from the review below.

Review:
{review_text}

{format_instructions}
""",
        input_variables=["review_text"],
        partial_variables={
            "format_instructions": parser.get_format_instructions()
        }
    )

    extraction_chain = extraction_prompt | llm | parser

    complaint_data = extraction_chain.invoke({
        "review_text": review_text
    })

    print("\nSTEP 1: STRUCTURED COMPLAINT DATA")
    print(complaint_data.model_dump_json(indent=2))

    # STEP 2: Generate a support ticket using Step 1's data
    ticket_prompt = ChatPromptTemplate.from_messages([
        (
            "system",
            "You are a support ticket writer. Create a concise "
            "internal support ticket using only the supplied "
            "structured information. Do not invent facts."
        ),
        (
            "human",
            """
Complaint: {complaint}
Product or feature: {product_or_feature}
Customer sentiment: {sentiment}

Write a clear title and a concise summary for the support team.
"""
        )
    ])

    ticket_chain = ticket_prompt | llm | StrOutputParser()

    ticket_summary = ticket_chain.invoke(
        complaint_data.model_dump()
    )

    print("\nSTEP 2: SUPPORT TICKET SUMMARY")
    print(ticket_summary)


if __name__ == "__main__":
    main()
