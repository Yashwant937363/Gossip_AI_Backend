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
from transformers import BlipProcessor, BlipForConditionalGeneration
import requests
from io import BytesIO
from PIL import Image
import os


# Load BLIP model
processor = BlipProcessor.from_pretrained("Salesforce/blip-image-captioning-base")
blipmodel = BlipForConditionalGeneration.from_pretrained("Salesforce/blip-image-captioning-base")


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

class ImageRequest(BaseModel):
    url:str

@app.post("/api/ai/analyze-image")
async def analyze_image(request:ImageRequest):
    try:
        url = request.url
        response = requests.get(url)
        response.raise_for_status()  # Raise an error for bad responses (e.g., 404)
        image = Image.open(BytesIO(response.content))

        # Process the image with BLIP
        inputs = processor(image,return_tensors="pt")
        outputs = blipmodel.generate(**inputs)
        caption = processor.decode(outputs[0], skip_special_tokens=True)
        
        return {"caption": caption}

    except Exception as e:
        print(str(e))
        return {"error": str(e)}

if __name__ == "__main__":
    host = os.getenv("HOST", "localhost")
    port = int(os.getenv("PORT", 8000))
    uvicorn.run(app, host=host, port=port)
