
from pydantic import BaseModel, Field
from langchain_core.tools import StructuredTool
from langchain_core.messages import ToolMessage
from langchain_ollama import ChatOllama


# STEP 1: Define and validate tool inputs
class ExpenseInput(BaseModel):
    total_amount: float = Field(
        gt=0,
        description="Total bill amount in rupees"
    )
    num_people: int = Field(
        gt=0,
        description="Number of people sharing the bill"
    )


# STEP 2: Calculate each person's share
def split_expense(total_amount: float, num_people: int) -> str:
    share = total_amount / num_people
    return (
        f"Total bill: ₹{total_amount:.2f}; "
        f"People: {num_people}; "
        f"Each person owes: ₹{share:.2f}"
    )


expense_tool = StructuredTool.from_function(
    func=split_expense,
    name="split_expense",
    description=(
        "Split a total bill amount equally among a specified "
        "number of people."
    ),
    args_schema=ExpenseInput
)


def main():
    # STEP 3: Call the tool manually
    manual_result = expense_tool.invoke({
        "total_amount": 2400,
        "num_people": 4
    })

    print("\nSTEP 1: MANUAL TOOL RESULT")
    print(manual_result)

    # STEP 4: Ask the model to use the same tool
    llm = ChatOllama(
        model="llama3.2:3b",
        temperature=0
    )

    llm_with_tools = llm.bind_tools([expense_tool])

    user_query = "Split ₹2400 between 4 friends."
    response = llm_with_tools.invoke(user_query)

    print("\nSTEP 2: MODEL'S SUGGESTED TOOL CALLS")
    print(response.tool_calls)

    messages = [
        ("human", user_query),
        response
    ]

    # STEP 5: Manually execute the suggested tool call
    for call in response.tool_calls:
        if call["name"] == "split_expense":
            result = expense_tool.invoke(call["args"])

            print("\nSTEP 3: BOUND-FLOW TOOL RESULT")
            print(result)

            messages.append(
                ToolMessage(
                    content=str(result),
                    tool_call_id=call["id"]
                )
            )

    # STEP 6: Generate the final answer
    final_response = llm_with_tools.invoke(messages)

    print("\nSTEP 4: FINAL ANSWER")
    print(final_response.content)

    # STEP 7: Compare both results
    print("\nSTEP 5: RESULT COMPARISON")
    print("Manual result:", manual_result)

    if response.tool_calls:
        print("Bound-flow result:", result)
        print("Results match:", manual_result == result)


if __name__ == "__main__":
    main()
