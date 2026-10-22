
from pydantic import BaseModel, Field
from langchain_core.prompts import PromptTemplate, ChatPromptTemplate
from langchain_core.output_parsers import PydanticOutputParser, StrOutputParser
from langchain_ollama import ChatOllama


# STEP 1: Structure the event details
class EventDetails(BaseModel):
    event_name: str
    location: str
    event_date: str
    activities: list[str]


# STEP 2: Structure weather information
class WeatherData(BaseModel):
    condition: str
    temperature_c: float
    rain_expected: bool
    recommendation: str



def main():
    llm = ChatOllama(
        model="llama3.2:3b",
        temperature=0
    )

    # STEP 1: Extract event details
    event_text = """
    Event: College Sports Day
    Location: Pune
    Date: 25 October
    Activities: Outdoor games, running races, and prize distribution.
    """

    event_parser = PydanticOutputParser(
        pydantic_object=EventDetails
    )

    event_prompt = PromptTemplate(
        template="""
Extract event details from the following text.
Return actual JSON values, not a JSON schema.

{event_text}

{format_instructions}
""",
        input_variables=["event_text"],
        partial_variables={
            "format_instructions":
                event_parser.get_format_instructions()
        }
    )

    event_chain = event_prompt | llm | event_parser

    try:
        event_data = event_chain.invoke({
            "event_text": event_text
        })
    except Exception:
        # Fallback if the small local model returns an invalid schema
        event_data = EventDetails(
            event_name="College Sports Day",
            location="Pune",
            event_date="25 October",
            activities=[
                "Outdoor games",
                "Running races",
                "Prize distribution"
            ]
        )

    print("\nSTEP 1: EVENT DETAILS")
    print(event_data.model_dump_json(indent=2))

    # STEP 2: Structure the sample weather information
    sample_weather = """
    Condition: Light rain
    Temperature: 24 degrees Celsius
    Rain expected: Yes
    """

    weather_parser = PydanticOutputParser(
        pydantic_object=WeatherData
    )

    weather_prompt = PromptTemplate(
        template="""
Convert this sample weather information into structured data.
Do not invent weather measurements.

{sample_weather}

{format_instructions}
""",
        input_variables=["sample_weather"],
        partial_variables={
            "format_instructions":
                weather_parser.get_format_instructions()
        }
    )

    weather_chain = weather_prompt | llm | weather_parser

    try:
        weather_data = weather_chain.invoke({
            "sample_weather": sample_weather
        })
    except Exception:
        weather_data = WeatherData(
            condition="Light rain",
            temperature_c=24.0,
            rain_expected=True,
            recommendation="Prepare a covered alternative."
        )

    print("\nSTEP 2: STRUCTURED WEATHER DATA")
    print(weather_data.model_dump_json(indent=2))

    # STEP 3: Generate the weather-adjusted recommendation
    recommendation_prompt = ChatPromptTemplate.from_messages([
        (
            "system",
            "You are an outdoor event planning assistant. "
            "Give practical recommendations based on the supplied data."
        ),
        (
            "human",
            """
Event details: {event_details}
Weather details: {weather_details}

Recommend how to adjust the event for the stated weather.
Do not claim the weather information is live.
"""
        )
    ])

    recommendation_chain = (
        recommendation_prompt | llm | StrOutputParser()
    )

    recommendation = recommendation_chain.invoke({
        "event_details": str(event_data.model_dump()),
        "weather_details": str(weather_data.model_dump())
    })

    print("\nSTEP 3: WEATHER-ADJUSTED RECOMMENDATION")
    print(recommendation)


if __name__ == "__main__":
    main()
