
from langchain_core.tools import tool
from langchain_ollama import ChatOllama


@tool
def find_restaurants(cuisine: str, city: str) -> str:
    """Find restaurants matching a cuisine in a city."""
    restaurants = {
        ("Italian", "Mumbai"): ["Little Italy", "Toscano"],
        ("Indian", "Mumbai"): ["Shree Thaker Bhojanalay", "Soam"],
        ("Chinese", "Mumbai"): ["Yauatcha", "Mainland China"],
    }
    results = restaurants.get(
        (cuisine.title(), city.title()),
        ["Restaurant A", "Restaurant B"]
    )
    return f"{cuisine} restaurants in {city}: {', '.join(results)}"


@tool
def get_restaurant_rating(restaurant_name: str) -> str:
    """Get the rating of a restaurant."""
    ratings = {
        "Little Italy": "4.2/5",
        "Toscano": "4.3/5",
        "Shree Thaker Bhojanalay": "4.4/5",
        "Soam": "4.2/5",
        "Yauatcha": "4.1/5",
        "Mainland China": "4.0/5",
    }
    return f"{restaurant_name} rating: {ratings.get(restaurant_name, 'Not available')}"


@tool
def check_restaurant_open(restaurant_name: str, day: str) -> str:
    """Check whether a restaurant is open on a given day."""
    return f"{restaurant_name} is listed as open on {day} (sample data)."


@tool
def estimate_cost_for_two(restaurant_name: str) -> str:
    """Estimate the meal cost for two people."""
    costs = {
        "Little Italy": "₹2,000",
        "Toscano": "₹2,500",
        "Shree Thaker Bhojanalay": "₹800",
        "Soam": "₹1,200",
        "Yauatcha": "₹3,000",
        "Mainland China": "₹1,800",
    }
    return f"Estimated cost for two at {restaurant_name}: {costs.get(restaurant_name, '₹1,500')}"


def main():
    llm = ChatOllama(model="llama3.2:3b", temperature=0)

    tools = [
        find_restaurants,
        get_restaurant_rating,
        check_restaurant_open,
        estimate_cost_for_two
    ]

    llm_with_tools = llm.bind_tools(tools)

    query = (
        "Recommend an Italian restaurant in Mumbai for dinner on Saturday. "
        "Find restaurants, get a rating, check whether one is open, "
        "and estimate the cost for two."
    )

    # STEP 1: Show the model's suggested tool calls
    print("\nSTEP 1: MODEL'S SUGGESTED TOOL CALLS")
    response = llm_with_tools.invoke(query)
    print(response.tool_calls)

    # STEP 2: Execute tools in the correct order
    print("\nSTEP 2: TOOL RESULTS")

    restaurant_result = find_restaurants.invoke({
        "cuisine": "Italian",
        "city": "Mumbai"
    })
    print("\nfind_restaurants:", restaurant_result)

    # Use an actual restaurant name from the returned list
    restaurant_name = "Little Italy"

    rating_result = get_restaurant_rating.invoke({
        "restaurant_name": restaurant_name
    })
    print("\nget_restaurant_rating:", rating_result)

    open_result = check_restaurant_open.invoke({
        "restaurant_name": restaurant_name,
        "day": "Saturday"
    })
    print("\ncheck_restaurant_open:", open_result)

    cost_result = estimate_cost_for_two.invoke({
        "restaurant_name": restaurant_name
    })
    print("\nestimate_cost_for_two:", cost_result)

    # STEP 3: Generate a final recommendation using real tool results
    context = (
        f"Restaurant search: {restaurant_result}\n"
        f"Rating: {rating_result}\n"
        f"Opening status: {open_result}\n"
        f"Estimated cost: {cost_result}"
    )

    final_response = llm.invoke([
        (
            "system",
            "You are a restaurant recommendation assistant. "
            "Use only the supplied tool results. Do not invent details. "
            "Give a concise plain-text recommendation, not code."
        ),
        (
            "human",
            f"Recommend a restaurant for Italian dinner in Mumbai on Saturday.\n"
            f"Use these tool results:\n{context}"
        )
    ])

    print("\nSTEP 3: FINAL RESTAURANT RECOMMENDATION")
    print(final_response.content or context)


if __name__ == "__main__":
    main()
