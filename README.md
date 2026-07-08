# SriLankan Airlines RAG Chatbot (SLAL-Chatbot)

A Retrieval-Augmented Generation (RAG) chatbot developed as part of an MSc assignment in **Natural Language Processing and Generative AI**.

The chatbot answers questions related to SriLankan Airlines policies by retrieving relevant information from a knowledge base created using official SriLankan Airlines documentation. It uses a Retrieval-Augmented Generation (RAG) pipeline to provide grounded responses while reducing hallucinations.

---

## Features

- Retrieval-Augmented Generation (RAG)
- FAISS vector database for semantic search
- Sentence Transformers embeddings (`all-MiniLM-L6-v2`)
- Google FLAN-T5 language model (`flan-t5-base`)
- Gradio web interface
- Evidence validation before answer generation
- Prompt injection protection
- Toxicity filtering
- Fallback responses for out-of-scope questions

---

## Technologies Used

- Python 3.11
- LangChain
- Hugging Face Transformers
- Sentence Transformers
- FAISS
- Gradio

---

## Repository Structure

```
SLAL_chatbot/
│
├── app.py                     # Gradio web application
├── chatbot.py                 # RAG pipeline and chatbot logic
├── build_index.py             # Creates the FAISS vector store
├── requirements.txt           # Project dependencies
├── SLAL_chatbot.ipynb         # Google Colab notebook
├── README.md
├── .gitignore
│
├── dataset/                   # Knowledge base documents (excluded from Git)
├── vector_store/              # Generated FAISS index (excluded from Git)
└── venv/                      # Python virtual environment (excluded from Git)
```

---

## Installation

Clone the repository:

```bash
git clone <repository-url>
cd SLAL_chatbot
```

Create and activate a virtual environment:

### macOS / Linux

```bash
python3.11 -m venv venv
source venv/bin/activate
```

### Windows

```bash
python -m venv venv
venv\Scripts\activate
```

Install the required packages:

```bash
pip install -r requirements.txt
```

---

## Dataset

Place all SriLankan Airlines knowledge base `.txt` documents inside the `dataset/` directory.

After adding or modifying documents, rebuild the vector database:

```bash
python build_index.py
```

---

## Running the Application

Start the chatbot using:

```bash
python app.py
```

A Gradio interface will open in your browser.

---

## Google Colab

The project also includes a Google Colab notebook (`SLAL_chatbot.ipynb`) that reproduces the complete RAG pipeline in a notebook environment.

---

## Safeguards

The chatbot includes multiple safeguards to improve reliability and reduce incorrect responses:

- Retrieval grounding using a FAISS vector store
- Evidence validation before response generation
- Prompt injection detection
- Toxicity filtering
- Fallback response for questions outside the knowledge base

---

## License

This project was developed for academic purposes as part of an MSc assignment.

The SriLankan Airlines documentation used to build the knowledge base was obtained from publicly available official sources and is intended solely for educational use.