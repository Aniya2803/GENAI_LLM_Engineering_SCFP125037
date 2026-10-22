
from langchain_core.tools import tool
from langchain_core.messages import ToolMessage
from langchain_ollama import ChatOllama


# Sample weather data
WEATHER = {
    "Mumbai": {
        "temperature": 29,
        "conditions": "Humid and partly cloudy"
    },
    "Pune": {
        "temperature": 24,
        "conditions": "Cloudy"
    }
}


# Tool 1: Get weather information
@tool
def get_weather(city: str) -> str:
    """Get sample current temperature and weather conditions for a city."""
    data = WEATHER.get(city)

    if not data:
        return f"Weather data is unavailable for {city}."

    return (
        f"City: {city}, Temperature: {data['temperature']}°C, "
        f"Conditions: {data['conditions']}"
    )


# Tool 2: Recommend an outfit
@tool
def recommend_outfit(temperature: float, conditions: str) -> str:
    """Recommend an outfit based on temperature in Celsius and conditions."""
    if temperature >= 30:
        outfit = "lightweight, breathable cotton clothes"
    elif temperature >= 24:
        outfit = "light clothing with a comfortable layer if needed"
    else:
        outfit = "a light jacket or warm layers"

    if "rain" in conditions.lower():
        outfit += ", plus a raincoat or umbrella"

    return f"Recommended outfit: {outfit}."



def main():
    llm = ChatOllama(
        model="llama3.2:3b",
        temperature=0
    )

    tools = [get_weather, recommend_outfit]
    tool_map = {t.name: t for t in tools}
    llm_with_tools = llm.bind_tools(tools)

    user_query = "What should I wear today in Mumbai?"

    # ROUND 1: Ask specifically for the weather tool
    messages = [
        (
            "system",
            "You must call get_weather first to obtain actual supplied "
            "weather data. Never guess temperature or conditions. "
            "Do not call recommend_outfit until weather data is returned."
        ),
        ("human", user_query)
    ]

    response = llm_with_tools.invoke(messages)

    print("\nSTEP 1: FIRST-ROUND TOOL CALLS")
    print(response.tool_calls)

    messages.append(response)

    # Execute the weather tool using Mumbai as the city
    weather_result = get_weather.invoke({"city": "Mumbai"})

    print("\nSTEP 2: FETCHED SAMPLE WEATHER DATA")
    print(weather_result)

    # Add the actual weather result to the conversation
    weather_call = next(
        (
            call for call in response.tool_calls
            if call["name"] == "get_weather"
        ),
        None
    )

    if weather_call:
        messages.append(
            ToolMessage(
                content=str(weather_result),
                tool_call_id=weather_call["id"]
            )
        )
    else:
        # Record the tool result even if the model ignored instructions
        messages.append(
            (
                "human",
                f"Weather tool result for Mumbai: {weather_result}"
            )
        )

    # ROUND 2: Request outfit recommendation using fetched weather
    messages.append((
        "human",
        "Now call recommend_outfit using the exact temperature and "
        "conditions from the weather result. Do not invent values."
    ))

    response = llm_with_tools.invoke(messages)

    print("\nSTEP 3: SECOND-ROUND TOOL CALLS")
    print(response.tool_calls)

    messages.append(response)

    for call in response.tool_calls:
        if call["name"] == "recommend_outfit":
            result = tool_map["recommend_outfit"].invoke(call["args"])

            print("\nSTEP 4: OUTFIT TOOL RESULT")
            print(result)

            messages.append(
                ToolMessage(
                    content=str(result),
                    tool_call_id=call["id"]
                )
            )

    final_response = llm_with_tools.invoke(messages)

    print("\nSTEP 5: FINAL OUTFIT RECOMMENDATION")
    print(final_response.content)


if __name__ == "__main__":
    main()
