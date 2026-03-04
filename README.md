## Key Features

- **Host your own document QA (RAG) web-UI**: Support multi-user login, organize your files in private/public collections, collaborate and share your favorite chat with others.

- **Organize your LLM & Embedding models**: Support both local LLMs & popular API providers (OpenAI, Azure, Ollama, Groq).

- **Hybrid RAG pipeline**: Sane default RAG pipeline with hybrid (full-text & vector) retriever and re-ranking to ensure best retrieval quality.

- **Multi-modal QA support**: Perform Question Answering on multiple documents with figures and tables support. Support multi-modal document parsing (selectable options on UI).

- **Advanced citations with document preview**: By default the system will provide detailed citations to ensure the correctness of LLM answers. View your citations (incl. relevant score) directly in the _in-browser PDF viewer_ with highlights. Warning when retrieval pipeline return low relevant articles.

- **Support complex reasoning methods**: Use question decomposition to answer your complex/multi-hop question. Support agent-based reasoning with `ReAct`, `ReWOO` and other agents.

- **Configurable settings UI**: You can adjust most important aspects of retrieval & generation process on the UI (incl. prompts).

- **Extensible**: Being built on Gradio, you are free to customize or add any UI elements as you like. Also, we aim to support multiple strategies for document indexing & retrieval. `GraphRAG` indexing pipeline is provided as an example.

## Installation

### System requirements

1. [Python](https://www.python.org/downloads/) >= 3.10
2. [Docker](https://www.docker.com/): optional, if you [install with Docker](#with-docker-recommended)
3. [Unstructured](https://docs.unstructured.io/open-source/installation/full-installation#full-installation) if you want to process files other than `.pdf`, `.html`, `.mhtml`, and `.xlsx` documents. Installation steps differ depending on your operating system. Please visit the link and follow the specific instructions provided there.

### With Docker (recommended)

1. Build the Docker image from this repository:

   ```bash
   # from the repository root
   docker build -t kso-rag:lite --target lite .
   docker build -t kso-rag:full --target full .
   docker build -t kso-rag:ollama --target ollama .
   ```

2. Run the image (example: `full` image):

   ```bash
   docker run \
     -e GRADIO_SERVER_NAME=0.0.0.0 \
     -e GRADIO_SERVER_PORT=7860 \
     -v ./kso_rag_data:/app/kso_rag_data \
     -p 7860:7860 -it --rm \
     kso-rag:full
   ```

3. We currently support and test two platforms: `linux/amd64` and `linux/arm64` (for newer Mac). You can specify the platform by passing `--platform` in the `docker run` command. For example:

   ```bash
   docker run \
     -e GRADIO_SERVER_NAME=0.0.0.0 \
     -e GRADIO_SERVER_PORT=7860 \
     -v ./kso_rag_data:/app/kso_rag_data \
     -p 7860:7860 -it --rm \
     --platform linux/arm64 \
     kso-rag:lite
   ```

4. Once everything is set up correctly, you can go to `http://localhost:7860/` to access the WebUI.

#### Faster builds when developing

- **Use BuildKit and cache:**  
  `DOCKER_BUILDKIT=1 docker build --ssh default -t kso-rag:full --target full .`  
  Pip cache is reused between builds.

- **Layer order:** The Dockerfile copies dependency files first and app code last. When you only change files outside `src/` (e.g. `launch.sh`, `app.py`), the heavy `pip install` layer stays cached. When you change code inside `src/core` or `src/ui`, that layer is invalidated and pip runs again.

- **Dev without rebuild:** To test code changes without rebuilding the image, use the dev compose file and volume mounts:
  ```bash
  docker compose -f docker-compose.dev.yml up --build   # first time
  docker compose -f docker-compose.dev.yml up            # later: code from host, no rebuild
  ```
  This mounts `./src`, `app.py`, and `launch.sh` into the container so edits on the host are used immediately (restart the app or the container if needed).

### Without Docker

#### Option 1: Using uv (Recommended for faster installation)

1. Clone the repository and run the uv installation script:

   ```shell
   # clone this repo (replace with your own repository URL)
   git clone https://github.com/YOUR_ORG/kso-rag
   cd kso-rag

   # run the uv installation script (installs uv automatically if not present)
   bash scripts/run_uv.sh
   ```

   This script will:
   - Install uv package manager if not present
   - Create a virtual environment with Python 3.10
   - Install all dependencies using uv (significantly faster than conda/pip)
   - Set up PDF.js viewer
   - Launch the application

#### Option 2: Manual Installation (Step-by-step)

**Important**: Install `pyparsing<3.1.0` first to avoid compatibility issues with `httplib2`.

1. Create a virtual environment:

   ```shell
   python3.10 -m venv .venv
   source .venv/bin/activate  # macOS/Linux
   # hoặc
   .venv\Scripts\activate  # Windows
   ```

2. Install dependencies in the correct order:

   ```shell
   # Install pyparsing first (required for httplib2 compatibility)
   pip install "pyparsing<3.1.0"
   
   # Install core package
   pip install -e "src/core[all]"
   
   # Install UI package
   pip install -e "src/ui"
   
   # Fix known version conflicts
   pip uninstall hnswlib chroma-hnswlib -y 2>/dev/null || true
   pip install chroma-hnswlib
   ```

3. (Optional) Setup PDF.js viewer:

   ```shell
   bash scripts/download_pdfjs.sh src/ui/kso_rag_ui/kso_rag_ui/assets/prebuilt/pdfjs-4.0.379-dist
   ```

4. Create a `.env` file in the root of this project. Use `.env.example` as a template

5. Run the application:

   ```shell
   python app.py
   ```

   The app will open automatically in your browser at `http://localhost:7860`

   **Default credentials:**
   - Username: `admin`
   - Password: `admin`

<!--
### Setup GraphRAG

> [!NOTE]
> Official MS GraphRAG indexing only works with OpenAI or Ollama API.
> We recommend most users to use NanoGraphRAG implementation for straightforward integration with kso-rag.

<details>

<summary>Setup Nano GRAPHRAG</summary>

- Install nano-GraphRAG: `pip install nano-graphrag`
- `nano-graphrag` install might introduce version conflicts (see project issues for workarounds)
  - To quickly fix: `pip uninstall hnswlib chroma-hnswlib && pip install chroma-hnswlib`
- Launch kso-rag with `USE_NANO_GRAPHRAG=true` environment variable.
- Set your default LLM & Embedding models in Resources setting and it will be recognized automatically from NanoGraphRAG.

</details>

<details>

<summary>Setup LIGHTRAG</summary>

- Install LightRAG: `pip install git+https://github.com/HKUDS/LightRAG.git`
- `LightRAG` install might introduce version conflicts (see project issues for workarounds)
  - To quickly fix: `pip uninstall hnswlib chroma-hnswlib && pip install chroma-hnswlib`
- Launch kso-rag with `USE_LIGHTRAG=true` environment variable.
- Set your default LLM & Embedding models in Resources setting and it will be recognized automatically from LightRAG.

</details>

<details>

<summary>Setup MS GRAPHRAG</summary>

- **Non-Docker Installation**: If you are not using Docker, install GraphRAG with the following command:

  ```shell
  pip install "graphrag<=0.3.6" future
  ```

- **Setting Up API KEY**: To use the GraphRAG retriever feature, ensure you set the `GRAPHRAG_API_KEY` environment variable. You can do this directly in your environment or by adding it to a `.env` file.
- **Using Local Models and Custom Settings**: If you want to use GraphRAG with local models (like `Ollama`) or customize the default LLM and other configurations, set the `USE_CUSTOMIZED_GRAPHRAG_SETTING` environment variable to true. Then, adjust your settings in the `settings.yaml.example` file.

</details>

### Setup Local Models (for local/private RAG)

See [Local model setup](docs/local_model.md).

### Setup multimodal document parsing (OCR, table parsing, figure extraction)

These options are available:

- [Azure Document Intelligence (API)](https://azure.microsoft.com/en-us/products/ai-services/ai-document-intelligence)
- [Adobe PDF Extract (API)](https://developer.adobe.com/document-services/docs/overview/pdf-extract-api/)
- [Docling (local, open-source)](https://github.com/DS4SD/docling)
  - To use Docling, first install required dependencies: `pip install docling`

Select corresponding loaders in `Settings -> Retrieval Settings -> File loader`

### Customize your application

- By default, all application data is stored in the `./kso_rag_data` folder. You can back up or copy this folder to transfer your installation to a new machine.

- For advanced users or specific use cases, you can customize these files:

  - `flowsettings.py`
  - `.env`

#### `flowsettings.py`

This file contains the configuration of your application. You can use the example
[here](flowsettings.py) as the starting point.

<details>

<summary>Notable settings</summary>

```python
# setup your preferred document store (with full-text search capabilities)
KSO_RAG_DOCSTORE=(Elasticsearch | LanceDB | SimpleFileDocumentStore)

# setup your preferred vectorstore (for vector-based search)
KSO_RAG_VECTORSTORE=(ChromaDB | LanceDB | InMemory | Milvus | Qdrant)

# Enable / disable multimodal QA
KSO_RAG_REASONINGS_USE_MULTIMODAL=True

# Setup your new reasoning pipeline or modify existing one.
KSO_RAG_REASONINGS = [
    "kso_rag_ui.reasoning.simple.FullQAPipeline",
    "kso_rag_ui.reasoning.simple.FullDecomposeQAPipeline",
    "kso_rag_ui.reasoning.react.ReactAgentPipeline",
    "kso_rag_ui.reasoning.rewoo.RewooAgentPipeline",
]
```

</details>

#### `.env`

This file provides another way to configure your models and credentials.

<details>

<summary>Configure model via the .env file</summary>

- Alternatively, you can configure the models via the `.env` file with the information needed to connect to the LLMs. This file is located in the folder of the application. If you don't see it, you can create one.

- Currently, the following providers are supported:

  - **OpenAI**

    In the `.env` file, set the `OPENAI_API_KEY` variable with your OpenAI API key in order
    to enable access to OpenAI's models. There are other variables that can be modified,
    please feel free to edit them to fit your case. Otherwise, the default parameter should
    work for most people.

    ```shell
    OPENAI_API_BASE=https://api.openai.com/v1
    OPENAI_API_KEY=<your OpenAI API key here>
    OPENAI_CHAT_MODEL=gpt-3.5-turbo
    OPENAI_EMBEDDINGS_MODEL=text-embedding-ada-002
    ```

  - **Azure OpenAI**

    For OpenAI models via Azure platform, you need to provide your Azure endpoint and API
    key. Your might also need to provide your developments' name for the chat model and the
    embedding model depending on how you set up Azure development.

    ```shell
    AZURE_OPENAI_ENDPOINT=
    AZURE_OPENAI_API_KEY=
    OPENAI_API_VERSION=2024-02-15-preview
    AZURE_OPENAI_CHAT_DEPLOYMENT=gpt-35-turbo
    AZURE_OPENAI_EMBEDDINGS_DEPLOYMENT=text-embedding-ada-002
    ```

  - **Local Models**

    - Using `ollama` OpenAI compatible server:

      - Install [ollama](https://github.com/ollama/ollama) and start the application.

      - Pull your model, for example:

        ```shell
        ollama pull llama3.1:8b
        ollama pull nomic-embed-text
        ```

      - Set the model names on web UI and make it as default:

        ![Models](docs/images/models.png)

    - Using `GGUF` with `llama-cpp-python`

      You can search and download a LLM to be ran locally from the [Hugging Face Hub](https://huggingface.co/models). Currently, these model formats are supported:

      - GGUF

        You should choose a model whose size is less than your device's memory and should leave
        about 2 GB. For example, if you have 16 GB of RAM in total, of which 12 GB is available,
        then you should choose a model that takes up at most 10 GB of RAM. Bigger models tend to
        give better generation but also take more processing time.

        Here are some recommendations and their size in memory:

      - [Qwen1.5-1.8B-Chat-GGUF](https://huggingface.co/Qwen/Qwen1.5-1.8B-Chat-GGUF/resolve/main/qwen1_5-1_8b-chat-q8_0.gguf?download=true): around 2 GB

        Add a new LlamaCpp model with the provided model name on the web UI.

  </details>
-->