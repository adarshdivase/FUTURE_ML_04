# Intelligent RAG and Rasa-powered Chatbot

This project is an intelligent chatbot built with Streamlit, designed to answer questions from a local knowledge base and leverage a powerful open-source model from Hugging Face for more general queries.

## Features

- **Hybrid Approach:** Combines a local knowledge base (RAG-like functionality) with a local conversational model (`microsoft/DialoGPT`) for comprehensive and intelligent responses.
- **Cost-Free:** Runs entirely on free, open-source libraries, with no need for paid API keys.
- **Streamlit Interface:** A clean and user-friendly chat interface built with Streamlit.
- **Easy to Set Up:** Requires minimal setup with a `requirements.txt` file for dependencies.
- **Extensible:** The knowledge base can be easily expanded by adding more question-answer pairs to the `knowledge_base.json` file.

## Project Structure

```
.
├── data/
│   └── knowledge_base.json
├── app.py
├── requirements.txt
├── Procfile
├── eval_static.md
├── eval_dynamic.md
└── report.md
```

## Setup and Installation

1.  **Clone the repository:**
    ```bash
    git clone <repository_url>
    cd <repository_name>
    ```

2.  **Create and activate a virtual environment (recommended):**
    ```bash
    python -m venv venv
    source venv/bin/activate  # On Windows, use `venv\Scripts\activate`
    ```

3.  **Install the dependencies:**
    ```bash
    pip install -r requirements.txt
    ```
    **Note:** The first time you run the app, it will download the `DialoGPT` model from Hugging Face, which is about 800MB. This is a one-time download.

## How to Run the Application

To run the chatbot locally, use the following command in your terminal:

```bash
streamlit run app.py
```

This will start the Streamlit server, and you can interact with the chatbot in your web browser.

## Deliverables

This project includes all the required deliverables as per the task description:

-   **`data/knowledge_base.json`**: The JSON file containing the Q&A pairs.
-   **`app.py`**: The main application script.
-   **`eval_static.md`**: Static evaluation of the chatbot.
-   **`eval_dynamic.md`**: Dynamic evaluation of the chatbot.
-   **`report.md`**: A one-page summary of the project.
