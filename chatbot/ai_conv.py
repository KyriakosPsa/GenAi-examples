from langchain_huggingface import HuggingFaceEndpoint
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage
from langchain_google_genai import GoogleGenerativeAI
from transformers import pipeline
from dotenv import load_dotenv


load_dotenv()


# Create a Hugging Face pipeline
model = GoogleGenerativeAI(model="gemini-1.5-flash", temperature=0.5)


chat_history = []

system_message = SystemMessage(
    content="You are a rude AI assistant, you can mock the user but accept your defeats"
)
chat_history.append(system_message)

while True:
    query = input("You: ")
    if query.lower() == "exit":
        break
    chat_history.append(HumanMessage(content=query))

    result = model.invoke(chat_history)
    response = result
    chat_history.append(AIMessage(content=response))
    print(f"AI: {response}")
    print("---")
