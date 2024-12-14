import json
import os
import requests
import dotenv
from openai import OpenAI

dotenv.load_dotenv()
client = OpenAI()


def get_current_temperature(latitude, longitude):
    """Get the current temperature in a given latitude and longitude"""

    request_url = f"https://api.open-meteo.com/v1/forecast?latitude={latitude}&longitude={longitude}&hourly=temperature_2m&forecast_days=1"
    response = requests.get(request_url)

    return response.content


def run_conversation(content, messages=[]):
    messages = messages
    system_message = """
You are TemperatureBot, a model designed to help answer the user with temperature related questions
"""
    if not messages:
        messages.append({"role": "system", "content": system_message})

    messages.append({"role": "user", "content": content})

    tools = [
        {
            "type": "function",
            "function": {
                "name": "get_current_temperature",
                "description": "Get the current weather in a given latitude and longitude",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "latitude": {
                            "type": "string",
                            "description": "The latitude of a place",
                        },
                        "longitude": {
                            "type": "string",
                            "description": "The longitude of a place",
                        },
                    },
                    "required": ["latitude", "longitude"],
                },
            },
        }
    ]
    response = client.chat.completions.create(
        model="gpt-3.5-turbo-0125",
        messages=messages,
        tools=tools,
        tool_choice="auto",
        function_call="auto",
    )
    response_message = response.choices[0].message

    tool_calls = response_message.tool_calls

    if tool_calls:
        available_functions = {
            "get_current_temperature": get_current_temperature,
        }
        for tool_call in tool_calls:
            function_name = tool_call.function.name
            function_to_call = available_functions[function_name]
            function_args = json.loads(tool_call.function.arguments)
            function_response = function_to_call(
                latitude=function_args.get("latitude"),
                longitude=function_args.get("longitude"),
            )
            function_response_json = json.loads(function_response.decode("utf-8"))
            function_response_str = json.dumps(function_response_json, indent=2)

            messages.append(
                {
                    "tool_call_id": tool_call.id,
                    "role": "tool",
                    "name": function_name,
                    "content": function_response_str,
                }
            )

        second_response = client.chat.completions.create(
            model="gpt-3.5-turbo-0125", messages=messages
        )
        messages.append({"role": "assistant", "content": second_response})
        return second_response.choices[0].message.content

    else:
        return response_message.content


if __name__ == "__main__":

    while True:
        question = input("You:")
        if question == "exit":
            break
        else:
            response = run_conversation(question)
            print(f"TemperatureBot:{response}")
