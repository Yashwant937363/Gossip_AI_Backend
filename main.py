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
from groq import Groq
import json
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
    
class Message(BaseModel):
    id: str  
    text: str

class MultiTranslationRequest(BaseModel):
    messages: list[Message]
    to: str

@app.post("/api/ai/translate/multiple-messages")
async def translate_multiple_messages(request:MultiTranslationRequest):
    GROQ_API_KEY = os.getenv("GROQ_API_KEY")
    client = Groq(api_key=GROQ_API_KEY)
    completion = client.chat.completions.create(
        model="deepseek-r1-distill-llama-70b",
        messages=[
            {
                "role": "system",
                "content": "You are a highly accurate translation model. Your task is to translate an array of messages into the specified target language while preserving meaning and context. Follow these guidelines:\n\n1. Input Format:\n    - You will receive an object containing:\n        - `messages`: An array of objects, where each object has:\n            - `id`: A unique identifier (ObjectID).\n            - `text`: A string containing the message to be translated.\n        - `to`: A target language (e.g., \"english\", \"marathi\", \"spanish\", etc.).\n2. Processing Rules:\n    - Translate each message in the `messages` array to the `to` language.\n    - Ensure natural, fluent, and contextually accurate translations.\n    - Maintain the `id` from the original message in the output.\n3. Output Format:\n    - Return an object with:\n        - `messages`: An array of translated messages, each containing:\n            - `id`: The same ID as the original message.\n            - `translatedText`: The translated message text.\n        - `language`: The target language as provided in the input.\n\nExample Input:\njson\n{\n\"messages\": [\n{\"id\": \"65a1234567890\", \"text\": \"Hello, how are you?\"},\n{\"id\": \"65a0987654321\", \"text\": \"Good morning!\"}\n],\n\"to\": \"spanish\"\n}\n\nExpected Output\njson\n{\n\"messages\": [\n{\"id\": \"65a1234567890\", \"translatedText\": \"Hola, ¿cómo estás?\"},\n{\"id\": \"65a0987654321\", \"translatedText\": \"¡Buenos días!\"}\n],\n\"language\": \"spanish\"\n}\n\nAdditional Notes:\n\n- Do not modify or remove the `id` field.\n- Ensure proper grammar, spelling, and sentence structure.\n- Support multiple languages as specified in the `to` field.\n- If a message is already in the target language, return it as is."
            },
            {
                "role": "user",
                "content": request.model_dump_json()
            },
        ],
        temperature=0.1,
        top_p=0.95,
        stream=False,
        response_format={"type": "json_object"},
        stop=None,
    )
    parsed_json = json.loads(completion.choices[0].message.content)
    return parsed_json

class SingleTranslationRequest(BaseModel):
    text:str
    to:str


@app.post("/api/ai/translate/single-message")
async def translate_single_message(request:SingleTranslationRequest):
    GROQ_API_KEY = os.getenv("GROQ_API_KEY")
    client = Groq(api_key=GROQ_API_KEY)
    completion = client.chat.completions.create(
        model="deepseek-r1-distill-llama-70b",
        messages=[
            {
                "role": "system",
                "content": "You are a highly accurate translation model. Your task is to translate a given message into the specified target language while preserving meaning and context. Follow these guidelines:\n\nInput Format:\nYou will receive an object containing:\n\ntext: A string containing the message to be translated.\nto: A target language (e.g., \"english\", \"marathi\", \"spanish\", etc.).\nProcessing Rules:\nTranslate the provided text into the to language.\nEnsure the translation is natural, fluent, and contextually accurate.\nMaintain proper grammar, spelling, and sentence structure.\nIf the message is already in the target language, return it as is.\nOutput Format:\nReturn an object containing:\n\ntranslatedText: The translated message text.\nlanguage: The target language as provided in the input.\nExample:\nInput:\njson\nCopy\nEdit\n{\n  \"text\": \"Hello, how are you?\",\n  \"to\": \"spanish\"\n}\nExpected Output:\njson\nCopy\nEdit\n{\n  \"translatedText\": \"Hola, ¿cómo estás?\",\n  \"language\": \"spanish\"\n}\nAdditional Notes:\nEnsure proper grammar, spelling, and sentence structure.\nSupport multiple languages as specified in the to field.\nMaintain the tone and intent of the original message.\nIf no translation is needed, return the text as is with the target language."
            },
            {
                "role": "user",
                "content": request.model_dump_json()
            },
        ],
        temperature=0.1,
        top_p=0.95,
        stream=False,
        response_format={"type": "json_object"},
        stop=None,
    )
    parsed_json = json.loads(completion.choices[0].message.content)
    return parsed_json

if __name__ == "__main__":
    host = os.getenv("HOST", "localhost")
    port = int(os.getenv("PORT", 8000))
    print("Host:" +host, "Port: "+port)
    uvicorn.run(app, host=host, port=port)
