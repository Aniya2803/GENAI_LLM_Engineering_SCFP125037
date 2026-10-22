
from langchain_core.tools import StructuredTool
from langchain_ollama import ChatOllama
from pydantic import BaseModel, Field


# Input schema for BMI
class BMIInput(BaseModel):
    weight_kg: float = Field(gt=0, description="Weight in kilograms")
    height_cm: float = Field(gt=0, description="Height in centimeters")


# Input schema for daily calories
class CaloriesInput(BaseModel):
    weight_kg: float = Field(gt=0)
    height_cm: float = Field(gt=0)
    age: int = Field(gt=0)
    activity_level: str = Field(
        description="Activity level: sedentary, moderate, or active"
    )


def calculate_bmi(weight_kg: float, height_cm: float) -> str:
    height_m = height_cm / 100
    bmi = weight_kg / (height_m ** 2)
    return f"BMI: {bmi:.2f}"


def calculate_daily_calories(
    weight_kg: float,
    height_cm: float,
    age: int,
    activity_level: str
) -> str:
    # Illustrative estimate using the Mifflin-St Jeor equation for males
    bmr = 10 * weight_kg + 6.25 * height_cm - 5 * age + 5

    multipliers = {
        "sedentary": 1.2,
        "moderate": 1.55,
        "active": 1.725
    }

    multiplier = multipliers.get(activity_level.lower(), 1.2)
    calories = bmr * multiplier

    return f"Estimated daily calories: {calories:.0f} kcal"


class HealthToolkit:
    def __init__(self):
        self.tools = [
            StructuredTool.from_function(
                func=calculate_bmi,
                name="calculate_bmi",
                description="Calculate BMI from weight in kg and height in cm.",
                args_schema=BMIInput
            ),
            StructuredTool.from_function(
                func=calculate_daily_calories,
                name="calculate_daily_calories",
                description="Estimate daily calorie needs from weight, height, age, and activity level.",
                args_schema=CaloriesInput
            )
        ]

    def get_tools(self):
        return self.tools


def main():
    llm = ChatOllama(model="llama3.2:3b", temperature=0)

    toolkit = HealthToolkit()
    tools = toolkit.get_tools()
    tool_map = {tool.name: tool for tool in tools}
    llm_with_tools = llm.bind_tools(tools)

    query = (
        "For a 60 kg, 165 cm tall, 23-year-old person with moderate activity, "
        "calculate both BMI and estimated daily calorie needs. "
        "You must call both calculate_bmi and calculate_daily_calories."
    )

    response = llm_with_tools.invoke(query)

    print("\nSTEP 1: MODEL'S TOOL CALLS")
    print(response.tool_calls)

    print("\nSTEP 2: TOOL RESULTS")
    messages = [("human", query), response]

    for call in response.tool_calls:
        selected_tool = tool_map.get(call["name"])
        if selected_tool:
            result = selected_tool.invoke(call["args"])
            print(f"{call['name']}: {result}")
            from langchain_core.messages import ToolMessage
            messages.append(
                ToolMessage(
                    content=str(result),
                    tool_call_id=call["id"]
                )
            )

    final_response = llm_with_tools.invoke(messages)

    print("\nSTEP 3: FINAL HEALTH METRICS")
    print(final_response.content)


if __name__ == "__main__":
    main()
