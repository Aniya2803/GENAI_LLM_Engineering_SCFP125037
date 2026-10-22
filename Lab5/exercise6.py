
from pydantic import BaseModel, Field
from langchain_core.prompts import PromptTemplate, ChatPromptTemplate
from langchain_core.output_parsers import PydanticOutputParser
from langchain_ollama import ChatOllama


# STEP 1: Extract the user's fitness goal
class FitnessGoal(BaseModel):
    goal_type: str
    fitness_level: str
    days_available: int
    equipment_access: str


# STEP 2: Structure the weekly workout plan
class WorkoutDay(BaseModel):
    day: str
    activity: str
    duration_minutes: int


class WeeklyWorkoutPlan(BaseModel):
    weekly_plan: list[WorkoutDay] = Field(
        description="Workout activities for each available day"
    )


def main():
    llm = ChatOllama(
        model="llama3.2:3b",
        temperature=0
    )

    fitness_goal_text = """
    I want to improve my general fitness.
    I am a beginner and can exercise 3 days per week.
    I have no gym equipment.
    """

    days_per_week = 3

    # STEP 1: Extract structured fitness information
    goal_parser = PydanticOutputParser(
        pydantic_object=FitnessGoal
    )

    goal_prompt = PromptTemplate(
        template="""
Extract the fitness goal, fitness level, days available,
and equipment access from this text.

Fitness goal:
{fitness_goal_text}

Days available per week: {days_per_week}

Return actual JSON data, not a JSON schema.
{format_instructions}
""",
        input_variables=["fitness_goal_text", "days_per_week"],
        partial_variables={
            "format_instructions":
                goal_parser.get_format_instructions()
        }
    )

    goal_chain = goal_prompt | llm | goal_parser

    try:
        goal_data = goal_chain.invoke({
            "fitness_goal_text": fitness_goal_text,
            "days_per_week": days_per_week
        })
    except Exception:
        goal_data = FitnessGoal(
            goal_type="Improve general fitness",
            fitness_level="Beginner",
            days_available=3,
            equipment_access="No equipment"
        )

    print("\nSTEP 1: FITNESS GOAL")
    print(goal_data.model_dump_json(indent=2))

    # STEP 2: Generate a structured weekly plan
    plan_parser = PydanticOutputParser(
        pydantic_object=WeeklyWorkoutPlan
    )

    plan_prompt = ChatPromptTemplate.from_messages([
        (
            "system",
            "You are a cautious fitness planning assistant. "
            "Create a simple beginner-friendly bodyweight plan. "
            "Use safe, moderate activities and allow rest days."
        ),
        (
            "human",
            """
Create a weekly workout plan based on these details:
{goal_data}

Create exactly {days_available} workout days.
Each day must include day, activity, and duration_minutes.
Keep the plan simple and suitable for a beginner.
Return actual JSON data, not a JSON schema.

{format_instructions}
"""
        )
    ])

    plan_chain = plan_prompt | llm | plan_parser

    try:
        weekly_plan = plan_chain.invoke({
            "goal_data": str(goal_data.model_dump()),
            "days_available": goal_data.days_available,
            "format_instructions":
                plan_parser.get_format_instructions()
        })
    except Exception:
        weekly_plan = WeeklyWorkoutPlan(
            weekly_plan=[
                WorkoutDay(
                    day="Monday",
                    activity="Easy walking and gentle stretching",
                    duration_minutes=20
                ),
                WorkoutDay(
                    day="Wednesday",
                    activity="Easy walking and beginner bodyweight exercises",
                    duration_minutes=20
                ),
                WorkoutDay(
                    day="Friday",
                    activity="Easy walking and gentle stretching",
                    duration_minutes=20
                )
            ]
        )

    print("\nSTEP 2: WEEKLY WORKOUT PLAN")
    print(weekly_plan.model_dump_json(indent=2))


if __name__ == "__main__":
    main()
