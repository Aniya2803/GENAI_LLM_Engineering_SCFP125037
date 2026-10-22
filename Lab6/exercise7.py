
from langchain_core.tools import tool
from langchain_core.messages import ToolMessage
from langchain_ollama import ChatOllama


# Tool 1: Estimate flight cost
@tool
def get_flight_cost(origin: str, destination: str) -> str:
    """Estimate round-trip flight cost per person between two cities."""
    prices = {
        ("Delhi", "Goa"): 8500,
        ("Goa", "Delhi"): 8500,
        ("Mumbai", "Goa"): 5000,
        ("Goa", "Mumbai"): 5000
    }

    cost = prices.get((origin, destination), 10000)
    return f"Estimated round-trip flight cost per person: ₹{cost}"


# Tool 2: Estimate hotel cost
@tool
def get_hotel_cost(city: str, nights: int) -> str:
    """Estimate hotel cost for the requested number of nights."""
    nightly_rates = {
        "Goa": 3000,
        "Delhi": 3500,
        "Mumbai": 4000
    }

    rate = nightly_rates.get(city, 3000)
    total = rate * nights
    return f"Estimated hotel cost in {city} for {nights} nights: ₹{total}"


# Tool 3: Estimate food budget
@tool
def get_food_budget(city: str, days: int) -> str:
    """Estimate food budget for the requested number of days."""
    daily_budgets = {
        "Goa": 800,
        "Delhi": 1000,
        "Mumbai": 1000
    }

    budget = daily_budgets.get(city, 800)
    total = budget * days
    return f"Estimated food budget in {city} for {days} days: ₹{total}"


def main():
    llm = ChatOllama(
        model="llama3.2:3b",
        temperature=0
    )

    tools = [
        get_flight_cost,
        get_hotel_cost,
        get_food_budget
    ]

    tool_map = {t.name: t for t in tools}
    llm_with_tools = llm.bind_tools(tools)

    user_query = (
        "I'm planning a 3-night trip from Delhi to Goa. "
        "Estimate round-trip flights, hotel for 3 nights, "
        "and food for 4 days."
    )

    # STEP 1: Ask the model to select the required tools
    response = llm_with_tools.invoke(user_query)

    print("\nSTEP 1: MODEL'S SUGGESTED TOOL CALLS")
    print(response.tool_calls)

    # STEP 2: Execute every suggested tool call
    messages = [
        ("human", user_query),
        response
    ]

    for call in response.tool_calls:
        selected_tool = tool_map.get(call["name"])

        if selected_tool:
            result = selected_tool.invoke(call["args"])

            print(f"\nRESULT FROM {call['name']}")
            print(result)

            messages.append(
                ToolMessage(
                    content=str(result),
                    tool_call_id=call["id"]
                )
            )

    # STEP 3: Ask the model for the combined estimate
    final_response = llm_with_tools.invoke(messages)

    print("\nSTEP 3: FINAL TRIP COST ESTIMATE")
    print(final_response.content)


if __name__ == "__main__":
    main()
