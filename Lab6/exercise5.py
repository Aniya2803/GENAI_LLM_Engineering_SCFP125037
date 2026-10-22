
import asyncio
from typing import Type

from pydantic import BaseModel, Field
from langchain_core.tools import BaseTool
from langchain_ollama import ChatOllama


# Input schema for the tool
class StockInput(BaseModel):
    product_id: str = Field(
        description="ID of the product to check"
    )


# Custom tool with synchronous and asynchronous methods
class ProductStockTool(BaseTool):
    name: str = "check_product_stock"
    description: str = "Check whether a product is in stock using its ID."
    args_schema: Type[BaseModel] = StockInput

    def _run(self, product_id: str) -> str:
        """Synchronous stock check."""
        stock = {
            "P1023": True,
            "P1024": False,
            "P1025": True
        }

        if product_id not in stock:
            return f"Product {product_id} was not found."

        status = "in stock" if stock[product_id] else "out of stock"
        return f"Product {product_id} is {status}."

    async def _arun(self, product_id: str) -> str:
        """Asynchronous stock check."""
        return self._run(product_id)


async def main():
    stock_tool = ProductStockTool()

    # STEP 1: Call the tool synchronously
    sync_result = stock_tool.invoke({"product_id": "P1023"})

    print("\nSTEP 1: SYNCHRONOUS TOOL RESULT")
    print(sync_result)

    # STEP 2: Call the tool asynchronously
    async_result = await stock_tool.ainvoke({
        "product_id": "P1023"
    })

    print("\nSTEP 2: ASYNCHRONOUS TOOL RESULT")
    print(async_result)

    # STEP 3: Bind the tool to the LLM
    llm = ChatOllama(
        model="llama3.2:3b",
        temperature=0
    )

    llm_with_tools = llm.bind_tools([stock_tool])

    user_query = "Is product P1023 in stock?"
    response = llm_with_tools.invoke(user_query)

    print("\nSTEP 3: MODEL'S SUGGESTED TOOL CALLS")
    print(response.tool_calls)

    # STEP 4: Execute the suggested tool call manually
    from langchain_core.messages import ToolMessage

    messages = [
        ("human", user_query),
        response
    ]

    for call in response.tool_calls:
        if call["name"] == stock_tool.name:
            result = stock_tool.invoke(call["args"])

            print("\nSTEP 4: EXECUTED TOOL RESULT")
            print(result)

            messages.append(
                ToolMessage(
                    content=str(result),
                    tool_call_id=call["id"]
                )
            )

    # STEP 5: Get the model's final answer
    final_response = llm_with_tools.invoke(messages)

    print("\nSTEP 5: FINAL ANSWER")
    print(final_response.content)


if __name__ == "__main__":
    asyncio.run(main())
