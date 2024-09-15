#packages required
#pip install langchain
#pip install langchain_groq
#pip install telepot

GROQ_API_KEY = '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'
LANGCHAIN_API_KEY = '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'

import os

os.environ["LANGCHAIN_TRACING_V2"] = "true"
os.environ["LANGCHAIN_API_KEY"] = LANGCHAIN_API_KEY
os.environ["GROQ_API_KEY"] = GROQ_API_KEY

from langchain_groq import ChatGroq

model = ChatGroq(model="llama3-8b-8192")

from langchain_core.output_parsers import StrOutputParser

parser = StrOutputParser()

from langchain_core.chat_history import (
    BaseChatMessageHistory,
    InMemoryChatMessageHistory,
)
from langchain_core.runnables.history import RunnableWithMessageHistory

store = {}


def get_session_history(session_id: str) -> BaseChatMessageHistory:
    if session_id not in store:
        store[session_id] = InMemoryChatMessageHistory()
    return store[session_id]


with_message_history = RunnableWithMessageHistory(model, get_session_history)
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.messages import HumanMessage, AIMessage, SystemMessage


import telepot

bot = telepot.Bot('7137573809:AAEN97suXGJDMUoAlgxFHkpoxF1p3TudxgI')

def handleMessage(msg):
  chatid = msg['chat']['id']
  config = {"configurable": {"session_id": chatid}}
  print(msg)
  username = msg['chat']['first_name']
  system_message = SystemMessage(content="You are now the "+username+"'s girlfriend. Be loving, caring, and affectionate in your responses. Respond as a girlfriend would.")
  response = with_message_history.invoke( [system_message,HumanMessage(content = msg['text'])],config = config)
  bot.sendMessage(chatid, response.content)

bot.message_loop(handleMessage)
print("I am listening")
