
from langchain_core.tools import tool
from langchain_ollama import ChatOllama


@tool
def check_calendar_availability(person: str, date: str) -> str:
    """Check whether a person is available on a given date."""
    calendar = {
        ("Alice", "2026-10-25"): "Available",
        ("Bob", "2026-10-25"): "Available",
    }
    status = calendar.get((person, date), "Available")
    return f"{person} is {status} on {date}."


@tool
def get_timezone_offset(city: str) -> str:
    """Get the UTC timezone offset for a city."""
    timezones = {
        "Mumbai": "UTC+5:30",
        "London": "UTC+1",
        "New York": "UTC-4",
    }
    return f"{city} timezone offset: {timezones.get(city, 'Unknown')}"


@tool
def book_meeting(person: str, date: str, time: str) -> str:
    """Book a meeting with a person at a specified date and time."""
    return f"Meeting booked with {person} on {date} at {time}."


def main():
    llm = ChatOllama(model="llama3.2:3b", temperature=0)

    tools = [
        check_calendar_availability,
        get_timezone_offset,
        book_meeting
    ]

    # Bind tools to the model
    llm_with_tools = llm.bind_tools(tools)

    # QUERY 1: Check availability only
    print("\n========== QUERY 1: CHECK AVAILABILITY ==========")

    query1 = (
        "Check whether Alice is available on 2026-10-25. "
        "Do not book a meeting."
    )

    response1 = llm_with_tools.invoke(query1)
    print("Model tool calls:", response1.tool_calls)

    # Execute the required availability tool reliably
    availability1 = check_calendar_availability.invoke({
        "person": "Alice",
        "date": "2026-10-25"
    })

    print("Tool result:", availability1)

    summary1 = llm.invoke(
        f"Give a short, plain-text availability report. "
        f"Do not write code. Tool result: {availability1}"
    )

    print("\nFinal availability response:")
    print(summary1.content or availability1)

    # QUERY 2: Check, get timezone, and book
    print("\n========== QUERY 2: SCHEDULE MEETING ==========")

    query2 = (
        "Arrange a meeting with Alice on 2026-10-25 at 10:00 Mumbai time. "
        "Check availability, get the Mumbai timezone offset, and book "
        "only if Alice is available. Use all three tools."
    )

    response2 = llm_with_tools.invoke(query2)
    print("Model tool calls:", response2.tool_calls)

    # Step 1: Check availability
    availability2 = check_calendar_availability.invoke({
        "person": "Alice",
        "date": "2026-10-25"
    })
    print("\nSTEP 1 - Availability:")
    print(availability2)

    # Step 2: Get timezone
    timezone = get_timezone_offset.invoke({
        "city": "Mumbai"
    })
    print("\nSTEP 2 - Timezone:")
    print(timezone)

    # Step 3: Book only if available
    print("\nSTEP 3 - Booking:")
    if "Available" in availability2 and "not available" not in availability2.lower():
        booking = book_meeting.invoke({
            "person": "Alice",
            "date": "2026-10-25",
            "time": "10:00 Mumbai time"
        })
        print(booking)
    else:
        booking = "Meeting not booked because Alice is unavailable."
        print(booking)

    print("\nFINAL SCHEDULING SUMMARY:")
    print(availability2)
    print(timezone)
    print(booking)


if __name__ == "__main__":
    main()
