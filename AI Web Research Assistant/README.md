# 🤖 AI Web Research Assistant

An **AI-powered Web Research Assistant** built using **Generative AI and
Retrieval-Augmented Generation (RAG)**. This application allows users to
input web page URLs, extract their content, and ask questions based on
the information from those pages.

The system processes the content, converts it into **vector
embeddings**, and retrieves the most relevant information to generate
accurate answers using a **Large Language Model (LLM)**.

------------------------------------------------------------------------

## 🚀 Features

-   Extracts and processes information from **multiple web URLs**
-   Uses **text chunking** for efficient document processing
-   Generates **vector embeddings** for semantic search
-   Implements **Retrieval-Augmented Generation (RAG)**
-   Provides **context-aware answers** using LLMs
-   Interactive **Streamlit-based UI**

------------------------------------------------------------------------

## 🖼️ Project Screenshots

### Application Interface

![Application UI](/Gemini_Generated_Image_ds0tkjds0tkjds0t)

### Generated Answer

![Answer Output](Gemini_Generated_Image_3xgpiz3xgpiz3xgp)

*(Add your screenshots inside an `images` folder in the repository)*

------------------------------------------------------------------------

## 🧠 How It Works

1.  User enters one or more **web URLs**
2.  The system extracts text from the web pages
3.  Content is **split into smaller chunks**
4.  Each chunk is converted into **vector embeddings**
5.  Embeddings are stored in a **vector database**
6.  When the user asks a question:
    -   The system retrieves relevant chunks
    -   Passes them to the **LLM**
    -   Generates an accurate response

------------------------------------------------------------------------

## 🛠 Tech Stack

-   Python
-   LangChain
-   OpenAI / LLM
-   FAISS (Vector Database)
-   Streamlit
-   BeautifulSoup
-   NumPy & Pandas

------------------------------------------------------------------------

## 📦 Installation

Clone the repository:

``` bash
git clone https://github.com/harisjirati/Python-GEN-AI.git
```

Navigate to the project folder:

``` bash
cd AI Web Research Assistant
```

Install dependencies:

``` bash
pip install -r requirements.txt
```

------------------------------------------------------------------------

## ▶️ Run the Application

``` bash
streamlit run main.py
```

After running the command, open the **Streamlit local server link** in
your browser.

------------------------------------------------------------------------

## 📌 Future Improvements

-   Add support for **PDF and document uploads**
-   Improve **retrieval accuracy with advanced RAG techniques**
-   Add **chat memory and conversation history**
-   Deploy the application on **cloud platforms**

------------------------------------------------------------------------

## 👨‍💻 Author

**Haris Jirati**\
Aspiring **Generative AI Engineer**

GitHub:\
https://github.com/harisjirati
