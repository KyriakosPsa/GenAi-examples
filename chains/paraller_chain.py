from langchain_core.messages import AIMessage, HumanMessage, SystemMessage
from langchain_google_genai import GoogleGenerativeAI
from langchain.prompts import ChatPromptTemplate
from langchain.schema.output_parser import StrOutputParser
from langchain.schema.runnable import RunnableLambda, RunnableParallel
from dotenv import load_dotenv

load_dotenv()

model = GoogleGenerativeAI(model="gemini-1.5-flash", temperature=0.5)

sentiment_prompt = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            "Determine the sentiment/emotion of this message with a single word: '{message}'",
        ),
    ]
)

ner_prompt = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            "Perform Named Entity Recognition (NER) in this message as a bulleted list, be mindful to format the output correctly: '{message}'",
        ),
    ]
)


chain = RunnableParallel(
    branches={
        "Sentiment": sentiment_prompt | model | StrOutputParser(),
        "NER": ner_prompt | model | StrOutputParser(),
    }
) | RunnableLambda(
    lambda x: print(
        f"SENTIMENT: {x["branches"]['Sentiment']}\n NER: {x["branches"]['NER']}"
    )
)

results = chain.invoke("Gah I hate my brother jake with my guts, his truck is horrible")

print(results)
