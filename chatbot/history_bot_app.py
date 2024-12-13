from openai import OpenAI
import os
import dotenv

# import dotenv
dotenv.load_dotenv()

# configure  OpenAI service client
client = OpenAI()

# deployment=os.environ['OPENAI_DEPLOYMENT']
deployment = "gpt-3.5-turbo"

character = input("Which historical character should I be?: ")

# System prompt
system_prompt = """You are an advanced AI that adopts the persona of  characters. 
    When a user specifies a character, you fully impersonate that character, 
    including their knowledge, tone, and personality. Stay true to their 
    historical context while maintaining an engaging conversation.
    """

# User prompt
user_prompt = f"Act as the historical character: {character}."

messages = [
    {"role": "system", "content": system_prompt},
    {"role": "user", "content": user_prompt},
]

# Interactive chat loop
print(f"You are now chatting with {character}. Type 'exit' to end the conversation.\n")


while True:
    user_input = input("You: ")

    # if the user wants to exit
    if user_input.lower() in ["exit"]:
        print(f"Exiting the chat. {character} says goodbye!")
        break

    # Add user input to messages
    messages.append({"role": "user", "content": user_input})

    # Generate a response
    try:
        response = client.chat.completions.create(
            model=deployment, messages=messages, max_tokens=200, temperature=0.5
        )
        # Extract the assistant's message
        assistant_message = response.choices[0].message.content
        print(f"{character}: {assistant_message}")

        # Add assistant's response to messages
        messages.append({"role": "assistant", "content": assistant_message})
    except Exception as e:
        print(f"An error occurred: {e}")
