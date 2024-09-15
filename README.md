# Gossip AI Server

This project is a FastAPI-based server that utilizes the LangChain framework and the ChatGroq model to create an AI-powered chatbot. The chatbot can respond in various tones such as friendly, formal, sarcastic, compassionate, or humorous. The AI is designed to act like a human assistant, responding to user input based on session-specific configurations.

## Features

- Multiple configurable chatbot tones: 
  - Friendly
  - Formal
  - Sarcastic
  - Compassionate
  - Humorous
- Maintains chat history using session IDs
- AI behaves like a human assistant in all interactions
- Easily configurable via API requests

## Technologies Used

- **FastAPI**: A modern, fast web framework for building APIs with Python.
- **LangChain**: A framework for developing applications powered by large language models.
- **ChatGroq**: The AI model used for generating responses.
- **uvicorn**: ASGI server for serving FastAPI applications.
- **Pydantic**: Data validation and parsing.
