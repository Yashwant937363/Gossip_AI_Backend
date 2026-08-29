# Gossip AI Backend

A FastAPI-based AI backend server providing intelligent conversational, translation, image analysis, and summarization services powered by Groq (Llama 3), Google Gemini, and OpenAI-compatible models.

## Features

### 1. **AI Chatbot with Tone Switching** (`/api/ai/chatbot`)

- Session-based conversational AI with in-memory chat history
- Supports 6 distinct tone modes:
  - **Normal** – Standard assistant responses
  - **Friendly** – Cheerful and casual
  - **Formal** – Professional and polite
  - **Sarcastic** – Witty and ironic
  - **Compassionate** – Empathetic and supportive (counselor-style)
  - **Humorous** – Comedic and lighthearted
- Powered by **Llama3-8b-8192** via Groq inference
- Maintains per-session conversation context using `InMemoryChatMessageHistory`

### 2. **Image Analysis** (`/api/ai/analyze-image`)

- Accepts a public image URL
- Downloads and processes the image using **Pillow**
- Sends the image to **Google Gemini 2.5 Flash** for content analysis
- Returns an AI-generated caption/description of the image

### 3. **Multi-Message Translation** (`/api/ai/translate/multiple-messages`)

- Batch-translates an array of messages into a target language
- Preserves original message IDs
- Supports any language (English, Marathi, Spanish, etc.)
- Returns structured JSON with `id` and `translatedText` per message
- Powered by **openai/gpt-oss-120b**

### 4. **Single Message Translation** (`/api/ai/translate/single-message`)

- Translates a single text message to a specified target language
- Maintains original tone and intent
- Returns the translated text and detected language
- Powered by **openai/gpt-oss-120b**

### 5. **Chat Summarization** (`/api/ai/summarize`)

- Summarizes structured conversations (array of username/message pairs)
- Three output formats:
  - **Paragraph** – Concise free-text summary
  - **Bullet** – Key points as bullet list
  - **Structured** – Organized with topic, key points, and conclusion
- Filters out greetings, fillers, and small talk
- Powered by **openai/gpt-oss-120b**

## Tech Stack

| Component                     | Technology                                  |
| ----------------------------- | ------------------------------------------- |
| **Framework**                 | FastAPI (Python)                            |
| **Web Server**                | Uvicorn                                     |
| **Chat Model**                | Groq – Llama3-8b-8192 (via LangChain)       |
| **Image Model**               | Google Gemini 2.5 Flash                     |
| **Translation/Summarization** | OpenAI-compatible – gpt-oss-120b (via Groq) |
| **Schema Validation**         | Pydantic v2                                 |
| **API Client**                | Requests (image download), Google GenAI SDK |

## Project Structure

```
├── main.py              # FastAPI application with all API endpoints
├── groqBot.py           # Standalone Telegram chatbot bot (legacy)
├── requirements.txt     # Python dependencies
├── .env                 # Environment variables (gitignored)
├── .env.example         # Environment variable template
├── .gitignore           # Git ignore rules
└── README.md            # Project documentation
```

## Environment Variables

Create a `.env` file in the project root:

```env
LANGCHAIN_TRACING_V2=true
GROQ_API_KEY=your_groq_api_key
LANGCHAIN_API_KEY=your_langchain_api_key
GOOGLE_API_KEY=your_google_genai_api_key
HOST=localhost          # Optional, defaults to localhost
PORT=8000               # Optional, defaults to 8000
```

> **Note:** The `.env` file is gitignored. Use `.env.example` as a template.

- **GROQ_API_KEY** – Required for chatbot and translation/summarization models (obtain from [console.groq.com](https://console.groq.com))
- **GOOGLE_API_KEY** – Required for image analysis (obtain from [Google AI Studio](https://aistudio.google.com/))
- **LANGCHAIN_API_KEY** – Required for LangSmith tracing (obtain from [smith.langchain.com](https://smith.langchain.com))

## Setup & Installation

### Prerequisites

- Python 3.10+
- pip

### Installation

```bash
# Clone the repository
git clone https://github.com/Yashwant937363/Gossip_AI_Backend.git
cd Gossip_AI_Backend

# Install dependencies
pip install -r requirements.txt

# Configure environment variables
cp .env.example .env
# Edit .env with your API keys
```

### Running the Server

```bash
python main.py
```

The server starts at `http://localhost:8000` by default. The host and port can be configured via the `HOST` and `PORT` environment variables.

### API Documentation

Once the server is running, interactive API documentation is available at:

- **Swagger UI:** [http://localhost:8000/docs](http://localhost:8000/docs)
- **ReDoc:** [http://localhost:8000/redoc](http://localhost:8000/redoc)

## API Endpoints

### 1. Chatbot

```
POST /api/ai/chatbot
```

**Request Body:**

```json
{
  "session_id": "user-123",
  "tone": "friendly",
  "message": "What's the weather like today?"
}
```

**Response:**

```json
{
  "response": "Hey! I'm not sure about today's weather specifically, but I hope it's a beautiful day wherever you are! ☀️"
}
```

### 2. Image Analysis

```
POST /api/ai/analyze-image
```

**Request Body:**

```json
{
  "url": "https://example.com/image.jpg"
}
```

**Response:**

```json
{
  "caption": "A detailed description of the image content..."
}
```

### 3. Translate Multiple Messages

```
POST /api/ai/translate/multiple-messages
```

**Request Body:**

```json
{
  "messages": [
    { "id": "65a1234567890", "text": "Hello, how are you?" },
    { "id": "65a0987654321", "text": "Good morning!" }
  ],
  "to": "spanish"
}
```

**Response:**

```json
{
  "messages": [
    { "id": "65a1234567890", "translatedText": "Hola, ¿cómo estás?" },
    { "id": "65a0987654321", "translatedText": "¡Buenos días!" }
  ],
  "language": "spanish"
}
```

### 4. Translate Single Message

```
POST /api/ai/translate/single-message
```

**Request Body:**

```json
{
  "text": "Hello, how are you?",
  "to": "spanish"
}
```

**Response:**

```json
{
  "translatedText": "Hola, ¿cómo estás?",
  "language": "spanish"
}
```

### 5. Summarize Chat

```
POST /api/ai/summarize
```

**Request Body:**

```json
{
  "conversation": [
    { "username": "Alice", "message": "I'm thinking of buying a new laptop." },
    { "username": "Bob", "message": "What's your budget?" },
    { "username": "Alice", "message": "Around $1000." }
  ],
  "method": "bullet"
}
```

**Response (bullet):**

```json
{
  "summary": [
    "Alice is looking for a new laptop.",
    "Budget: around $1000.",
    "Bob asks about budget."
  ]
}
```

**Response (paragraph):**

```json
{
  "summary": "Alice is looking for a new laptop with an approximate budget of $1000. Bob inquires about the budget to help with recommendations."
}
```

**Response (structured):**

```json
{
  "summary": {
    "topic": "Laptop Purchase",
    "keyPoints": ["Alice needs a new laptop.", "Budget: ~$1000."],
    "conclusion": "Alice is gathering information before making a purchase decision."
  }
}
```

## Architecture & Design

### Request Flow

```
Client → FastAPI → Route Handler → LLM Provider → JSON Response
```

### Session Management

- Chat history is stored **in-memory** using a dictionary of `InMemoryChatMessageHistory` objects
- Session IDs act as keys to maintain isolated conversations per user
- **Warning:** In-memory storage is not persistent and will be lost on server restart. For production, consider using a persistent store (e.g., Redis, database).

### Model Routing

| Service        | Model Provider    | Model Name            |
| -------------- | ----------------- | --------------------- |
| Chatbot        | Groq (LangChain)  | `llama3-8b-8192`      |
| Image Analysis | Google Gemini     | `gemini-2.5-flash`    |
| Translation    | Groq (OpenAI API) | `openai/gpt-oss-120b` |
| Summarization  | Groq (OpenAI API) | `openai/gpt-oss-120b` |

### Prompt Engineering

Each endpoint uses carefully crafted system prompts to guide the AI's behavior:

- **Chatbot** combines a "human behavior" system prompt with tone-specific instructions
- **Translation** prompts specify exact input/output JSON schemas with examples
- **Summarization** prompts include different output templates for each format style
- All prompts enforce **JSON-structured output** with `response_format={"type": "json_object"}` for translation and summarization endpoints

## Legacy Components

### Telegram Bot (`groqBot.py`)

A standalone Telegram chatbot using the **telepot** library:

- Responds to messages as the user's "girlfriend" persona
- Maintains per-chat session history
- **Note:** This is a legacy utility script with hardcoded API keys. It is not part of the main FastAPI server and should not be used in production without key rotation.

## Security Notes

- **API keys in `groqBot.py`** are hardcoded – do not commit this file to public repositories
- The main server reads API keys from the `.env` file, which is properly gitignored
- All endpoints accept `POST` requests with JSON bodies – ensure proper CORS configuration and authentication are added before production deployment
- In-memory session storage is ephemeral; implement persistent storage for production use

## Deployment

### Docker (Recommended)

```bash
# Build the image
docker build -t gossip-ai-backend .

# Run the container
docker run -d \
  --name gossip-ai-backend \
  -p 8000:8000 \
  --env-file .env \
  gossip-ai-backend
```

### Direct Deployment

```bash
pip install -r requirements.txt
python main.py
```

### Environment Configuration for Production

Set the following environment variables:

- `HOST` – Bind address (e.g., `0.0.0.0` for containerized deployments)
- `PORT` – Server port (default: `8000`)

## Development

### Adding a New Tone

Edit the `tone_templates` dictionary in `main.py`:

```python
tone_templates = {
    # ... existing tones ...
    "motivational": "Switch to a motivational tone. Inspire and encourage the user.",
}
```

### Adding a New Feature

1. Define a new Pydantic model for request validation
2. Create a new route handler with `@app.post("/api/ai/your-endpoint")`
3. Implement the AI logic using the appropriate model provider
4. Return a JSON-serializable response

## License

This project is provided as-is for educational and development purposes.
