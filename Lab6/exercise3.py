
from langchain_core.tools import tool
from langchain_ollama import ChatOllama


# Tool 1: Temperature conversion
@tool
def convert_temperature(value: float, conversion: str) -> str:
    """Convert temperature between Celsius and Fahrenheit.
    Use conversion='C_to_F' or conversion='F_to_C'."""
    if conversion == "C_to_F":
        result = (value * 9 / 5) + 32
        return f"{value}°C = {result:.2f}°F"
    elif conversion == "F_to_C":
        result = (value - 32) * 5 / 9
        return f"{value}°F = {result:.2f}°C"
    else:
        return "Invalid conversion. Use C_to_F or F_to_C."


# Tool 2: Distance conversion
@tool
def convert_distance(value: float, conversion: str) -> str:
    """Convert distance between kilometres and miles.
    Use conversion='km_to_miles' or conversion='miles_to_km'."""
    if conversion == "km_to_miles":
        result = value * 0.621371
        return f"{value} km = {result:.2f} miles"
    elif conversion == "miles_to_km":
        result = value / 0.621371
        return f"{value} miles = {result:.2f} km"
    else:
        return "Invalid conversion. Use km_to_miles or miles_to_km."


def main():
    llm = ChatOllama(
        model="llama3.2:3b",
        temperature=0
    )

    tools = [convert_temperature, convert_distance]
    llm_with_tools = llm.bind_tools(tools)

    user_query = "Convert 100 km to miles."

    # STEP 1: Ask the model to choose the correct tool
    first_response = llm_with_tools.invoke(user_query)

    print("\nSTEP 1: MODEL'S SUGGESTED TOOL CALLS")
    print(first_response.tool_calls)

    # STEP 2: Execute the suggested tool manually
    from langchain_core.messages import ToolMessage

    messages = [
        ("human", user_query),
        first_response
    ]

    for call in first_response.tool_calls:
        selected_tool = next(
            (t for t in tools if t.name == call["name"]),
            None
        )

        if selected_tool:
            result = selected_tool.invoke(call["args"])
            print("\nSTEP 2: EXECUTED TOOL RESULT")
            print(result)

            messages.append(
                ToolMessage(
                    content=str(result),
                    tool_call_id=call["id"]
                )
            )

    # STEP 3: Generate the final answer
    final_response = llm_with_tools.invoke(messages)

    print("\nSTEP 3: FINAL ANSWER")
    print(final_response.content)


if __name__ == "__main__":
    main()
