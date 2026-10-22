
from langchain_core.tools import tool
from langchain_ollama import ChatOllama


# Sample order database
ORDERS = {
    "ORD-4521": "Shipped and arriving tomorrow",
    "ORD-4522": "Processing",
    "ORD-4523": "Delivered"
}


# Define the order lookup tool
@tool
def get_order_status(order_id: str) -> str:
    """Look up the shipping status of an order using its order ID."""
    return ORDERS.get(order_id, "Order ID not found")


def main():
    llm = ChatOllama(
        model="llama3.2:3b",
        temperature=0
    )

    # Bind the tool to the model
    llm_with_tools = llm.bind_tools([get_order_status])

    user_query = "Where is my order ORD-4521?"

    # First model call: ask the model to select a tool
    first_response = llm_with_tools.invoke(user_query)

    print("\nSTEP 1: MODEL'S SUGGESTED TOOL CALLS")
    print(first_response.tool_calls)

    # Manually execute the suggested tool call
    tool_results = []

    for tool_call in first_response.tool_calls:
        if tool_call["name"] == "get_order_status":
            result = get_order_status.invoke(tool_call["args"])
            tool_results.append((tool_call, result))
            print("\nSTEP 2: MANUALLY EXECUTED TOOL RESULT")
            print(result)

    # Send the tool result back to the model
    messages = [
        ("human", user_query),
        first_response
    ]

    for tool_call, result in tool_results:
        from langchain_core.messages import ToolMessage

        messages.append(
            ToolMessage(
                content=str(result),
                tool_call_id=tool_call["id"]
            )
        )

    final_response = llm_with_tools.invoke(messages)

    print("\nSTEP 3: FINAL NATURAL-LANGUAGE ANSWER")
    print(final_response.content)


if __name__ == "__main__":
    main()
