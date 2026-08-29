import base64

from dotenv import load_dotenv
from langchain_groq import ChatGroq
from langchain_core.chat_history import (
    BaseChatMessageHistory,
    InMemoryChatMessageHistory,
)
from langchain_core.runnables.history import RunnableWithMessageHistory
from langchain_core.messages import HumanMessage, SystemMessage
import uvicorn
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import requests
from io import BytesIO
from PIL import Image
from groq import Groq
import json
import os
from google import genai # type: ignore
from google.genai import types # type: ignore
from io import BytesIO
from typing import List, Literal

load_dotenv()

# Store for message based chat history
store = {}

GROQ_API_KEY = os.getenv("GROQ_API_KEY")
client = Groq(api_key=GROQ_API_KEY)


def get_session_history(session_id: str) -> BaseChatMessageHistory:
    if session_id not in store:
        store[session_id] = InMemoryChatMessageHistory()
    return store[session_id]


# Define the model
model = ChatGroq(model="openai/gpt-oss-20b", max_tokens=256)
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
    print(response)
    return {"response": response}

class ImageRequest(BaseModel):
    url:str

@app.post("/api/ai/analyze-image")
async def analyze_image(request: ImageRequest):
    try:
        response = requests.get(request.url)
        response.raise_for_status()

        image_bytes = response.content

        # Convert image bytes to base64
        image_base64 = base64.b64encode(image_bytes).decode("utf-8")

        # Get MIME type from response
        content_type = response.headers.get("content-type", "image/jpeg")

        image_data_url = (
            f"data:{content_type};base64,{image_base64}"
        )

        SYSTEM_PROMPT = """
        You are an AI image analysis system.

        Analyze the provided image and return ONLY valid JSON.

        Use exactly this structure:

        {
        "title": "short name of the main subject",
        "description": "detailed description of what is visible in the image",
        "details": [
            "important visual detail",
            "important visual detail",
            "important visual detail"
        ],
        "mood": "short description of the mood or feeling conveyed by the image"
        }

        Rules:
        - Return ONLY JSON.
        - Do not use Markdown.
        - Do not use ```json.
        - Do not add any text before or after the JSON.
        - Only describe information that can reasonably be determined from the image.
        - Keep the description detailed but concise.
        - Add as many useful details as necessary to the details array.
        - If mood is not applicable, use an empty string.
        """


        result = client.chat.completions.create(
            model="qwen/qwen3.6-27b",
            messages=[
                {
                    "role": "system",
                    "content": SYSTEM_PROMPT,
                },
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "text",
                            "text": "Analyze this image and return the result using the required JSON structure.",
                        },
                        {
                            "type": "image_url",
                            "image_url": {
                                "url": image_data_url,
                            },
                        },
                    ],
                },
            ],
            temperature=0.2,
            reasoning_effort="default",
            stream=False,
            response_format={"type": "json_object"},
        )
        content = result.choices[0].message.content

        if content == None:
            raise ValueError("Value can't be none")
            

        content = result.choices[0].message.content

        if(content == None):
            raise ValueError("Value should not be None")

        analysis = json.loads(content)

        return {
            "data": analysis
        }

    except Exception as e:
        print(e)
        return {
            "error": str(e)
        }
    
class Message1(BaseModel):
    id: str  
    text: str

class MultiTranslationRequest(BaseModel):
    messages: list[Message1]
    to: str

@app.post("/api/ai/translate/multiple-messages")
async def translate_multiple_messages(request:MultiTranslationRequest):
    completion = client.chat.completions.create(
        model="openai/gpt-oss-120b",
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
    message = completion.choices[0].message.content

    if message is None:
        raise HTTPException(status_code=500, detail="No response from model")

    try:
        parsed_json = json.loads(message)
    except json.JSONDecodeError:
        raise HTTPException(
            status_code=500,
            detail=f"Model returned invalid JSON: {message}"
        )

    return parsed_json
class SingleTranslationRequest(BaseModel):
    text:str
    to:str


@app.post("/api/ai/translate/single-message")
async def translate_single_message(request:SingleTranslationRequest):
    completion = client.chat.completions.create(
        model="openai/gpt-oss-120b",
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
    message = completion.choices[0].message.content

    if message is None:
        raise HTTPException(status_code=500, detail="No response from model")

    try:
        parsed_json = json.loads(message)
    except json.JSONDecodeError:
        raise HTTPException(
            status_code=500,
            detail=f"Model returned invalid JSON: {message}"
        )

    return parsed_json
class Message2(BaseModel):
    username:str
    message:str


class SummarizeChatRequest(BaseModel):
    conversation:list[Message2]
    method:Literal["paragraph", "bullet", "structured"]

@app.post("/api/ai/summarize")
async def summerize_chat(request:SummarizeChatRequest):
    input_system_prompt = "You are an AI assistant designed to summarize structured conversations provided in JSON format. Your goal is to extract key details and generate a concise summary. Focus on capturing important topics, decisions made, and key exchanges while removing unnecessary small talk. Identify the main themes of the conversation and present the summary in the requested format. Maintain clarity and coherence, ensuring that the summarized content is easy to understand and accurately reflects the conversation.\n**Input Format:**\n\n- The conversation is provided as a JSON object under the `\"conversation\"` key.\n- Each message contains a `\"username\"` (who sent it) and a `\"message\"` (the text content).\n\n**Processing Guidelines:**\n\n1. **Identify Key Topics:** Determine the main subject(s) discussed.\n2. **Extract Key Points:** Capture relevant details, such as questions, suggestions, and decisions.\n3. **Summarize Concisely:** Remove greetings, fillers, and unnecessary exchanges while maintaining context.\n\nExample Input:\n{\n\"conversation\": [\n{\n\"username\": \"Person A\",\n\"message\": \"Hey, I’m thinking of buying a new laptop. Any recommendations?\"\n},\n{\n\"username\": \"Person B\",\n\"message\": \"What’s your budget and main use?\"\n},\n{\n\"username\": \"Person A\",\n\"message\": \"Around $1000. I’ll use it for programming and occasional gaming.\"\n},\n{\n\"username\": \"Person B\",\n\"message\": \"You should check out the Dell XPS 13 or the ASUS ROG Zephyrus G14.\"\n},\n{\n\"username\": \"Person A\",\n\"message\": \"That sounds good! I’ll research both models. Thanks!\"\n},\n{\n\"username\": \"Person B\",\n\"message\": \"No problem! Let me know if you need more help.\"\n}\n]\n}"
    output_bullet_points = "Summarize the conversation in bullet points. Focus on key topics discussed, important details, and decisions made. Ensure that each bullet point is clear and concise. Do not include small talk or unnecessary details. Use simple and informative statements.\nExample Structure of JSON:\n{\n\"summery\": [\n\"Concise key points summarizing the conversation.\",\n\"Each point should capture an essential detail or decision made.\"\n],\n}\nExample:\n{\n\"summary\": [\n\"Person A is looking for a new laptop.\",\n\"Budget: $1000, use: programming & gaming.\",\n\"Person B suggests Dell XPS 13 and ASUS ROG Zephyrus G14.\",\n\"Person A will research the suggested options before making a decision.\",\n\"Person B offers further assistance if needed.\"\n]\n}"
    output_paragraph =  "Summarize the conversation in a short, well-structured paragraph. Focus on the main topic, key\ndetails, and final decisions. Maintain readability and coherence, ensuring the summary is\ninformative while remaining concise. Avoid unnecessary small talk.\nExample Structure JSON:\n{\n”summery”:”Person A is looking for a new laptop with a $1000 budget, primarily for programming and occasional gaming. Person B suggests two models: Dell XPS 13 and ASUS ROG Zephyrus G14. Person A decides to research both options before making a purchase. Person B offers further assistance if needed.”\n}\nExample:\n{\n\"summary\": \"Person A is looking for a new laptop with a $1000 budget, primarily for programming and occasional gaming. Person B suggests two models: Dell XPS 13 and ASUS ROG Zephyrus G14. Person A decides to research both options before making a purchase. Person B offers further assistance if needed.\"\n}"
    output_structured = "Summarize the conversation in a structured format, including a topic, key points, and a conclusion. The topic should reflect the main subject of the conversation. The key points should capture the essential details, and the conclusion should summarize the final decision or outcome. Keep the summary precise and informative.\nExample Structure JSON:\n{\n”summery”:\n{\n\"topic\": \"Main subject of the conversation.\",\n\"keyPoints\": [\n\"Essential details extracted from the conversation.\",\n\"Key decisions or exchanges.\"\n],\n\"conclusion\": \"Final outcome or decision reached in the conversation.\"\n}\n}\nExample\n{\n\"summary\": {\n\"topic\": \"Laptop Recommendation\",\n\"keyPoints\": [\n\"Person A needs a laptop for programming & gaming (Budget: $1000).\",\n\"Person B suggests Dell XPS 13 and ASUS ROG Zephyrus G14.\",\n\"Person A decides to research before purchasing.\"\n],\n\"conclusion\": \"Person A will explore both laptop options before making a decision, with Person B available for further guidance.\"\n}\n}"

    output_system_prompt = ""
    if(request.method == "bullet"):
        output_system_prompt = output_bullet_points
    elif(request.method == 'paragraph'):
        output_system_prompt = output_paragraph
    elif(request.method == 'structured'):
        output_system_prompt = output_structured
    

    # Ensure input is a valid JSON-formatted string
    input = {
        "conversation": [msg.dict() for msg in request.conversation]
    }
    input_str = json.dumps(input)  # ✅ Convert to string

    completion = client.chat.completions.create(
        model="openai/gpt-oss-120b",
        messages=[
            {
                "role": "system",
                "content": f"{input_system_prompt}\n\n{output_system_prompt}"
            },
            {
                "role": "user",
                "content": input_str  # ✅ Pass stringified content
            }
        ],
        temperature=0.2,
        top_p=0.95,
        stream=False,
        response_format={"type": "json_object"},
        stop=None
    )

    message = completion.choices[0].message.content

    if message is None:
        raise HTTPException(status_code=500, detail="No response from model")

    try:
        parsed_json = json.loads(message)
    except json.JSONDecodeError:
        raise HTTPException(
            status_code=500,
            detail=f"Model returned invalid JSON: {message}"
        )

    return parsed_json
    



if __name__ == "__main__":
    host = os.getenv("HOST", "localhost")
    port = int(os.getenv("PORT", 8000))
    print("Host:" +host, "Port: ",port)
    uvicorn.run(app, host=host, port=port)