from langchain_core.messages import AIMessage, HumanMessage, SystemMessage
from langchain_google_genai import GoogleGenerativeAI
from langchain.prompts import ChatPromptTemplate
from langchain.schema.output_parser import StrOutputParser
from langchain.schema.runnable import RunnableLambda
from dotenv import load_dotenv


load_dotenv()

model = GoogleGenerativeAI(model="gemini-1.5-flash", temperature=0.5)

prompt_template = ChatPromptTemplate.from_messages(
    [
        ("system", "You are a {attribute} assistant"),
        ("human", "hello nerd, what is 1+1?"),
    ]
)

uppercase_output = RunnableLambda(lambda x: x.upper())
count_words = RunnableLambda(lambda x: f"Word count: {len(x.split(" "))} \n {x}")

chain = prompt_template | model | StrOutputParser() | uppercase_output | count_words


result = chain.invoke({"attribute": "super serious and strict"})

print(result)
exit()
