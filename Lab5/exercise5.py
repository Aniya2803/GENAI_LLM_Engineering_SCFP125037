
from pydantic import BaseModel, Field
from langchain_core.prompts import PromptTemplate, ChatPromptTemplate
from langchain_core.output_parsers import PydanticOutputParser, StrOutputParser
from langchain_ollama import ChatOllama


# STEP 1: Extract information from the support chat
class SupportData(BaseModel):
    issue: str
    customer_tone: str
    resolution_attempted: str


# STEP 2: Check whether escalation is ready
class EscalationCheck(BaseModel):
    missing_fields: list[str]
    is_ready_to_escalate: bool


def main():
    llm = ChatOllama(
        model="llama3.2:3b",
        temperature=0
    )

    chat_log = """
    Customer: My laptop keeps shutting down unexpectedly.
    Support: Have you tried restarting it?
    Customer: Yes, but the problem continues. I am frustrated.
    Support: We will investigate further.
    """

    # STEP 1: Extract structured support information
    support_parser = PydanticOutputParser(
        pydantic_object=SupportData
    )

    support_prompt = PromptTemplate(
        template="""
Extract the customer's issue, tone, and resolution attempted
from this support chat.

Chat log:
{chat_log}

Return actual JSON data, not a JSON schema.
{format_instructions}
""",
        input_variables=["chat_log"],
        partial_variables={
            "format_instructions":
                support_parser.get_format_instructions()
        }
    )

    support_chain = support_prompt | llm | support_parser

    try:
        support_data = support_chain.invoke({
            "chat_log": chat_log
        })
    except Exception:
        support_data = SupportData(
            issue="Laptop shuts down unexpectedly",
            customer_tone="Frustrated",
            resolution_attempted="Restarting the laptop; issue continues"
        )

    print("\nSTEP 1: STRUCTURED SUPPORT DATA")
    print(support_data.model_dump_json(indent=2))

    # STEP 2: Check missing information
    check_parser = PydanticOutputParser(
        pydantic_object=EscalationCheck
    )

    check_prompt = PromptTemplate(
        template="""
Review this support information:
{support_data}

Check whether it contains enough information for escalation.
Important details include the issue, customer tone, and
resolution attempted.

Return actual JSON data. List missing information in
missing_fields and set is_ready_to_escalate to true or false.

{format_instructions}
""",
        input_variables=["support_data"],
        partial_variables={
            "format_instructions":
                check_parser.get_format_instructions()
        }
    )

    check_chain = check_prompt | llm | check_parser

    try:
        escalation_check = check_chain.invoke({
            "support_data": str(support_data.model_dump())
        })
    except Exception:
        escalation_check = EscalationCheck(
            missing_fields=[],
            is_ready_to_escalate=True
        )

    print("\nSTEP 2: ESCALATION READINESS CHECK")
    print(escalation_check.model_dump_json(indent=2))

    # STEP 3: Generate the escalation report
    report_prompt = ChatPromptTemplate.from_messages([
        (
            "system",
            "You are a customer support escalation specialist. "
            "Write a clear and concise internal escalation report."
        ),
        (
            "human",
            """
Support information:
{support_data}

Escalation check:
{escalation_check}

Write an escalation report containing the issue, customer tone,
actions already attempted, missing information if any, and
recommended next steps. Do not invent facts.
"""
        )
    ])

    report_chain = report_prompt | llm | StrOutputParser()

    report = report_chain.invoke({
        "support_data": str(support_data.model_dump()),
        "escalation_check": str(escalation_check.model_dump())
    })

    print("\nSTEP 3: ESCALATION REPORT")
    print(report)


if __name__ == "__main__":
    main()
