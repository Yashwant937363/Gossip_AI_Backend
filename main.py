from dotenv import load_dotenv
from langchain_groq import ChatGroq
from langchain_core.chat_history import (
    BaseChatMessageHistory,
    InMemoryChatMessageHistory,
)
from langchain_core.runnables.history import RunnableWithMessageHistory
from langchain_core.messages import HumanMessage, SystemMessage
import uvicorn
from fastapi import FastAPI
from pydantic import BaseModel

load_dotenv()

# Store for message based chat history
store = {}


def get_session_history(session_id: str) -> BaseChatMessageHistory:
    if session_id not in store:
        store[session_id] = InMemoryChatMessageHistory()
    return store[session_id]


# Define the model
model = ChatGroq(model="llama3-8b-8192", max_tokens=50)
with_message_history = RunnableWithMessageHistory(model, get_session_history)


# Initialize the FastAPI app
app = FastAPI(title="Gossip AI Server")

# General SystemMessage to act like a human assistant
human_behavior_message = SystemMessage(
    content="From now on, you will act as a human assistant in all interactions. You will respond as if you are an actual person."
)

# Define a dictionary to map tones to SystemMessages
tone_templates = {
    "normal": "Respond as a normal assistant.",
    "friendly": "Please switch your tone to be more friendly. Respond as if you're a cheerful and casual assistant.",
    "formal": "Switch to a formal tone. I’d like responses to be professional and polite.",
    "sarcastic": "Change your tone to sarcastic. I want your replies to be witty and ironic.",
    "compassionate": "Act like a Compassionate Counselor. I need empathy and supportive responses.",
    "humorous": "Switch to a humorous tone. Respond like a comedian and make me laugh!"
}


class ChatRequest(BaseModel):
    session_id: str
    tone: str
    message: str


@app.post(path="/api/ai/chatbot",)
async def chatbot_route(request: ChatRequest):
    session_id = request.session_id
    tone = request.tone.lower()
    message = request.message

    system_message_content = tone_templates.get(
        tone, "Respond as a normal assistant.")
    system_message = SystemMessage(content=system_message_content)

    human_message = HumanMessage(content=message)
    response = with_message_history.invoke(
        [human_behavior_message, system_message, human_message], config={
            "configurable": {"session_id": session_id}}
    )

    return {"response": response}

if __name__ == "__main__":
    uvicorn.run(app, host="localhost", port=8000)
