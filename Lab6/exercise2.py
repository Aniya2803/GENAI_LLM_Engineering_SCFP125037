
from pydantic import BaseModel, Field
from langchain_core.tools import StructuredTool
from langchain_ollama import ChatOllama


# Define and validate the tool's input fields
class EMIInput(BaseModel):
    principal: float = Field(gt=0, description="Loan amount")
    annual_rate: float = Field(
        ge=0,
        le=30,
        description="Annual interest rate in percentage, from 0 to 30"
    )
    tenure_months: int = Field(
        gt=0,
        description="Loan tenure in months"
    )


# Function to calculate monthly EMI
def calculate_emi(
    principal: float,
    annual_rate: float,
    tenure_months: int
) -> str:
    monthly_rate = annual_rate / 12 / 100

    if monthly_rate == 0:
        emi = principal / tenure_months
    else:
        emi = (
            principal
            * monthly_rate
            * (1 + monthly_rate) ** tenure_months
            / ((1 + monthly_rate) ** tenure_months - 1)
        )

    return f"Monthly EMI: ₹{emi:.2f}"


# Create a StructuredTool with the Pydantic schema
emi_tool = StructuredTool.from_function(
    func=calculate_emi,
    name="calculate_emi",
    description=(
        "Calculate monthly loan EMI using principal, annual interest "
        "rate in percentage, and tenure in months."
    ),
    args_schema=EMIInput
)


def main():
    llm = ChatOllama(
        model="llama3.2:3b",
        temperature=0
    )

    llm_with_tools = llm.bind_tools([emi_tool])

    user_query = (
        "What's my EMI for a 5 lakh loan at 9% annual interest "
        "for 24 months?"
    )

    # STEP 1: Ask the model to select and fill the tool
    first_response = llm_with_tools.invoke(user_query)

    print("\nSTEP 1: MODEL'S SUGGESTED TOOL CALLS")
    print(first_response.tool_calls)

    # STEP 2: Execute the suggested tool manually
    messages = [
        ("human", user_query),
        first_response
    ]

    from langchain_core.messages import ToolMessage

    for call in first_response.tool_calls:
        if call["name"] == "calculate_emi":
            result = emi_tool.invoke(call["args"])

            print("\nSTEP 2: EXECUTED TOOL RESULT")
            print(result)

            messages.append(
                ToolMessage(
                    content=str(result),
                    tool_call_id=call["id"]
                )
            )

    # STEP 3: Ask the model for the final answer
    final_response = llm_with_tools.invoke(messages)

    print("\nSTEP 3: FINAL ANSWER")
    print(final_response.content)


if __name__ == "__main__":
    main()
