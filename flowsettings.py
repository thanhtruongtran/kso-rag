import os
from importlib.metadata import version
from inspect import currentframe, getframeinfo
from pathlib import Path

from decouple import config
from kso_rag_ui.utils.lang import SUPPORTED_LANGUAGE_MAP
from theflow.settings.default import *  # noqa

cur_frame = currentframe()
if cur_frame is None:
    raise ValueError("Cannot get the current frame.")
this_file = getframeinfo(cur_frame).filename
this_dir = Path(this_file).parent

# change this if your app use a different name
KSO_RAG_PACKAGE_NAME = "kso-rag-app"

# GitHub repo for your own build (e.g. "your-org/kso-rag"). Used for docs, changelogs, templates.
# Leave empty to use local docs only and skip remote links.
KSO_RAG_GITHUB_REPO = config("KSO_RAG_GITHUB_REPO", default="")

# Tunnel base URL for promptui share (e.g. "https://{appname}.promptui.dm.example.com").
# Used when generating the share link; set to your own tunnel endpoint.
KSO_RAG_TUNNEL_BASE_URL = config("KSO_RAG_TUNNEL_BASE_URL", default="")

KSO_RAG_APP_VERSION = config("KSO_RAG_APP_VERSION", None)
if not KSO_RAG_APP_VERSION:
    try:
        # Caution: This might produce the wrong version
        # https://stackoverflow.com/a/59533071
        KSO_RAG_APP_VERSION = version(KSO_RAG_PACKAGE_NAME)
    except Exception:
        KSO_RAG_APP_VERSION = "local"

KSO_RAG_GRADIO_SHARE = config("KSO_RAG_GRADIO_SHARE", default=False, cast=bool)
KSO_RAG_ENABLE_FIRST_SETUP = config(
    "KSO_RAG_ENABLE_FIRST_SETUP", default=True, cast=bool
)
KSO_RAG_DEMO_MODE = config("KSO_RAG_DEMO_MODE", default=False, cast=bool)
KSO_RAG_OLLAMA_URL = config("KSO_RAG_OLLAMA_URL", default="http://localhost:11434/v1/")

# App can be ran from anywhere and it's not trivial to decide where to store app data.
# So let's use the same directory as the flowsetting.py file.
KSO_RAG_APP_DATA_DIR = this_dir / "kso_rag_data"
KSO_RAG_APP_DATA_EXISTS = KSO_RAG_APP_DATA_DIR.exists()
KSO_RAG_APP_DATA_DIR.mkdir(parents=True, exist_ok=True)

# User data directory
KSO_RAG_USER_DATA_DIR = KSO_RAG_APP_DATA_DIR / "user_data"
KSO_RAG_USER_DATA_DIR.mkdir(parents=True, exist_ok=True)

# markdown output directory
KSO_RAG_MARKDOWN_OUTPUT_DIR = KSO_RAG_APP_DATA_DIR / "markdown_cache_dir"
KSO_RAG_MARKDOWN_OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# chunks output directory
KSO_RAG_CHUNKS_OUTPUT_DIR = KSO_RAG_APP_DATA_DIR / "chunks_cache_dir"
KSO_RAG_CHUNKS_OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# zip output directory
KSO_RAG_ZIP_OUTPUT_DIR = KSO_RAG_APP_DATA_DIR / "zip_cache_dir"
KSO_RAG_ZIP_OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# zip input directory
KSO_RAG_ZIP_INPUT_DIR = KSO_RAG_APP_DATA_DIR / "zip_cache_dir_in"
KSO_RAG_ZIP_INPUT_DIR.mkdir(parents=True, exist_ok=True)

# HF models can be big, let's store them in the app data directory so that it's easier
# for users to manage their storage.
# ref: https://huggingface.co/docs/huggingface_hub/en/guides/manage-cache
os.environ["HF_HOME"] = str(KSO_RAG_APP_DATA_DIR / "huggingface")
os.environ["HF_HUB_CACHE"] = str(KSO_RAG_APP_DATA_DIR / "huggingface")

# doc directory
KSO_RAG_DOC_DIR = this_dir / "docs"

KSO_RAG_MODE = "dev"
KSO_RAG_SSO_ENABLED = config("KSO_RAG_SSO_ENABLED", default=False, cast=bool)

KSO_RAG_FEATURE_CHAT_SUGGESTION = config(
    "KSO_RAG_FEATURE_CHAT_SUGGESTION", default=False, cast=bool
)
KSO_RAG_FEATURE_USER_MANAGEMENT = config(
    "KSO_RAG_FEATURE_USER_MANAGEMENT", default=True, cast=bool
)
KSO_RAG_USER_CAN_SEE_PUBLIC = None
KSO_RAG_FEATURE_USER_MANAGEMENT_ADMIN = str(
    config("KSO_RAG_FEATURE_USER_MANAGEMENT_ADMIN", default="admin")
)
KSO_RAG_FEATURE_USER_MANAGEMENT_PASSWORD = str(
    config("KSO_RAG_FEATURE_USER_MANAGEMENT_PASSWORD", default="admin")
)
KSO_RAG_ENABLE_ALEMBIC = False
KSO_RAG_DATABASE = f"sqlite:///{KSO_RAG_USER_DATA_DIR / 'sql.db'}"
KSO_RAG_FILESTORAGE_PATH = str(KSO_RAG_USER_DATA_DIR / "files")
KSO_RAG_WEB_SEARCH_BACKEND = (
    "kso_rag_core.indices.retrievers.tavily_web_search.WebSearch"
    # "kso_rag_core.indices.retrievers.jina_web_search.WebSearch"
)

KSO_RAG_DOCSTORE = {
    # "__type__": "kso_rag_core.storages.ElasticsearchDocumentStore",
    # "__type__": "kso_rag_core.storages.SimpleFileDocumentStore",
    "__type__": "kso_rag_core.storages.LanceDBDocumentStore",
    "path": str(KSO_RAG_USER_DATA_DIR / "docstore"),
}
# Vector store: Chroma (default) or Milvus. Set MILVUS_URI in .env to use Milvus.
# - Local file: MILVUS_URI=milvus.db
# - Milvus server: MILVUS_URI=http://localhost:19530
_MILVUS_URI = config("MILVUS_URI", default="")
if _MILVUS_URI:
    KSO_RAG_VECTORSTORE = {
        "__type__": "kso_rag_core.storages.MilvusVectorStore",
        "uri": _MILVUS_URI,
        "path": str(KSO_RAG_USER_DATA_DIR / "vectorstore"),
    }
else:
    KSO_RAG_VECTORSTORE = {
        # "__type__": "kso_rag_core.storages.LanceDBVectorStore",
        "__type__": "kso_rag_core.storages.ChromaVectorStore",
        # "__type__": "kso_rag_core.storages.MilvusVectorStore",
        # "__type__": "kso_rag_core.storages.QdrantVectorStore",
        "path": str(KSO_RAG_USER_DATA_DIR / "vectorstore"),
    }
KSO_RAG_LLMS = {}
KSO_RAG_EMBEDDINGS = {}
KSO_RAG_RERANKINGS = {}

# populate options from config
if config("AZURE_OPENAI_API_KEY", default="") and config(
    "AZURE_OPENAI_ENDPOINT", default=""
):
    if config("AZURE_OPENAI_CHAT_DEPLOYMENT", default=""):
        KSO_RAG_LLMS["azure"] = {
            "spec": {
                "__type__": "kso_rag_core.llms.AzureChatOpenAI",
                "temperature": 0,
                "azure_endpoint": config("AZURE_OPENAI_ENDPOINT", default=""),
                "api_key": config("AZURE_OPENAI_API_KEY", default=""),
                "api_version": config("OPENAI_API_VERSION", default="")
                or "2024-02-15-preview",
                "azure_deployment": config("AZURE_OPENAI_CHAT_DEPLOYMENT", default=""),
                "timeout": 20,
            },
            "default": False,
        }
    if config("AZURE_OPENAI_EMBEDDINGS_DEPLOYMENT", default=""):
        KSO_RAG_EMBEDDINGS["azure"] = {
            "spec": {
                "__type__": "kso_rag_core.embeddings.AzureOpenAIEmbeddings",
                "azure_endpoint": config("AZURE_OPENAI_ENDPOINT", default=""),
                "api_key": config("AZURE_OPENAI_API_KEY", default=""),
                "api_version": config("OPENAI_API_VERSION", default="")
                or "2024-02-15-preview",
                "azure_deployment": config(
                    "AZURE_OPENAI_EMBEDDINGS_DEPLOYMENT", default=""
                ),
                "timeout": 10,
            },
            "default": False,
        }

OPENAI_DEFAULT = "<YOUR_OPENAI_KEY>"
OPENAI_API_KEY = config("OPENAI_API_KEY", default=OPENAI_DEFAULT)
GOOGLE_API_KEY = config("GOOGLE_API_KEY", default="your-key")
IS_OPENAI_DEFAULT = len(OPENAI_API_KEY) > 0 and OPENAI_API_KEY != OPENAI_DEFAULT

if OPENAI_API_KEY:
    KSO_RAG_LLMS["openai"] = {
        "spec": {
            "__type__": "kso_rag_core.llms.ChatOpenAI",
            "temperature": 0,
            "base_url": config("OPENAI_API_BASE", default="")
            or "https://api.openai.com/v1",
            "api_key": OPENAI_API_KEY,
            "model": config("OPENAI_CHAT_MODEL", default="gpt-4o-mini"),
            "timeout": 20,
        },
        "default": IS_OPENAI_DEFAULT,
    }
    KSO_RAG_EMBEDDINGS["openai"] = {
        "spec": {
            "__type__": "kso_rag_core.embeddings.OpenAIEmbeddings",
            "base_url": config("OPENAI_API_BASE", default="https://api.openai.com/v1"),
            "api_key": OPENAI_API_KEY,
            "model": config(
                "OPENAI_EMBEDDINGS_MODEL", default="text-embedding-3-large"
            ),
            "timeout": 10,
            "context_length": 8191,
        },
        "default": IS_OPENAI_DEFAULT,
    }

VOYAGE_API_KEY = config("VOYAGE_API_KEY", default="")
if VOYAGE_API_KEY:
    KSO_RAG_EMBEDDINGS["voyageai"] = {
        "spec": {
            "__type__": "kso_rag_core.embeddings.VoyageAIEmbeddings",
            "api_key": VOYAGE_API_KEY,
            "model": config("VOYAGE_EMBEDDINGS_MODEL", default="voyage-3-large"),
        },
        "default": False,
    }
    KSO_RAG_RERANKINGS["voyageai"] = {
        "spec": {
            "__type__": "kso_rag_core.rerankings.VoyageAIReranking",
            "model_name": "rerank-2",
            "api_key": VOYAGE_API_KEY,
        },
        "default": False,
    }

if config("LOCAL_MODEL", default=""):
    KSO_RAG_LLMS["ollama"] = {
        "spec": {
            "__type__": "kso_rag_core.llms.ChatOpenAI",
            "base_url": KSO_RAG_OLLAMA_URL,
            "model": config("LOCAL_MODEL", default="qwen2.5:7b"),
            "api_key": "ollama",
        },
        "default": False,
    }
    KSO_RAG_LLMS["ollama-long-context"] = {
        "spec": {
            "__type__": "kso_rag_core.llms.LCOllamaChat",
            "base_url": KSO_RAG_OLLAMA_URL.replace("v1/", ""),
            "model": config("LOCAL_MODEL", default="qwen2.5:7b"),
            "num_ctx": 8192,
        },
        "default": False,
    }

    KSO_RAG_EMBEDDINGS["ollama"] = {
        "spec": {
            "__type__": "kso_rag_core.embeddings.OpenAIEmbeddings",
            "base_url": KSO_RAG_OLLAMA_URL,
            "model": config("LOCAL_MODEL_EMBEDDINGS", default="nomic-embed-text"),
            "api_key": "ollama",
        },
        "default": False,
    }
    KSO_RAG_EMBEDDINGS["fast_embed"] = {
        "spec": {
            "__type__": "kso_rag_core.embeddings.FastEmbedEmbeddings",
            "model_name": "BAAI/bge-base-en-v1.5",
        },
        "default": False,
    }

# additional LLM configurations
KSO_RAG_LLMS["claude"] = {
    "spec": {
        "__type__": "kso_rag_core.llms.chats.LCAnthropicChat",
        "model_name": "claude-3-5-sonnet-20240620",
        "api_key": "your-key",
    },
    "default": False,
}
KSO_RAG_LLMS["google"] = {
    "spec": {
        "__type__": "kso_rag_core.llms.chats.LCGeminiChat",
        "model_name": "gemini-1.5-flash",
        "api_key": GOOGLE_API_KEY,
    },
    "default": not IS_OPENAI_DEFAULT,
}
KSO_RAG_LLMS["groq"] = {
    "spec": {
        "__type__": "kso_rag_core.llms.ChatOpenAI",
        "base_url": "https://api.groq.com/openai/v1",
        "model": "llama-3.1-8b-instant",
        "api_key": "your-key",
    },
    "default": False,
}
KSO_RAG_LLMS["cohere"] = {
    "spec": {
        "__type__": "kso_rag_core.llms.chats.LCCohereChat",
        "model_name": "command-r-plus-08-2024",
        "api_key": config("COHERE_API_KEY", default="your-key"),
    },
    "default": False,
}
KSO_RAG_LLMS["mistral"] = {
    "spec": {
        "__type__": "kso_rag_core.llms.ChatOpenAI",
        "base_url": "https://api.mistral.ai/v1",
        "model": "ministral-8b-latest",
        "api_key": config("MISTRAL_API_KEY", default="your-key"),
    },
    "default": False,
}

# additional embeddings configurations
KSO_RAG_EMBEDDINGS["cohere"] = {
    "spec": {
        "__type__": "kso_rag_core.embeddings.LCCohereEmbeddings",
        "model": "embed-multilingual-v3.0",
        "cohere_api_key": config("COHERE_API_KEY", default="your-key"),
        "user_agent": "default",
    },
    "default": False,
}
KSO_RAG_EMBEDDINGS["google"] = {
    "spec": {
        "__type__": "kso_rag_core.embeddings.LCGoogleEmbeddings",
        "model": "models/text-embedding-004",
        "google_api_key": GOOGLE_API_KEY,
    },
    "default": not IS_OPENAI_DEFAULT,
}
KSO_RAG_EMBEDDINGS["mistral"] = {
    "spec": {
        "__type__": "kso_rag_core.embeddings.LCMistralEmbeddings",
        "model": "mistral-embed",
        "api_key": config("MISTRAL_API_KEY", default="your-key"),
    },
    "default": False,
}
# KSO_RAG_EMBEDDINGS["huggingface"] = {
#     "spec": {
#         "__type__": "kso_rag_core.embeddings.LCHuggingFaceEmbeddings",
#         "model_name": "sentence-transformers/all-mpnet-base-v2",
#     },
#     "default": False,
# }

# default reranking models
KSO_RAG_RERANKINGS["cohere"] = {
    "spec": {
        "__type__": "kso_rag_core.rerankings.CohereReranking",
        "model_name": "rerank-multilingual-v2.0",
        "cohere_api_key": config("COHERE_API_KEY", default=""),
    },
    "default": True,
}

KSO_RAG_REASONINGS = [
    "kso_rag_ui.reasoning.simple.FullQAPipeline",
    "kso_rag_ui.reasoning.simple.FullDecomposeQAPipeline",
    "kso_rag_ui.reasoning.react.ReactAgentPipeline",
    "kso_rag_ui.reasoning.rewoo.RewooAgentPipeline",
]
KSO_RAG_REASONINGS_USE_MULTIMODAL = config("USE_MULTIMODAL", default=False, cast=bool)
KSO_RAG_VLM_ENDPOINT = (
    "{0}/openai/deployments/{1}/chat/completions?api-version={2}".format(
        config("AZURE_OPENAI_ENDPOINT", default=""),
        config("OPENAI_VISION_DEPLOYMENT_NAME", default="gpt-4o"),
        config("OPENAI_API_VERSION", default=""),
    )
)


SETTINGS_APP: dict[str, dict] = {}


SETTINGS_REASONING = {
    "use": {
        "name": "Reasoning options",
        "value": None,
        "choices": [],
        "component": "radio",
    },
    "lang": {
        "name": "Language",
        "value": "en",
        "choices": [(lang, code) for code, lang in SUPPORTED_LANGUAGE_MAP.items()],
        "component": "dropdown",
    },
    "max_context_length": {
        "name": "Max context length (LLM)",
        "value": 32000,
        "component": "number",
    },
}

USE_GLOBAL_GRAPHRAG = config("USE_GLOBAL_GRAPHRAG", default=True, cast=bool)
USE_NANO_GRAPHRAG = config("USE_NANO_GRAPHRAG", default=False, cast=bool)
USE_LIGHTRAG = config("USE_LIGHTRAG", default=True, cast=bool)
USE_MS_GRAPHRAG = config("USE_MS_GRAPHRAG", default=True, cast=bool)

GRAPHRAG_INDEX_TYPES = []

if USE_MS_GRAPHRAG:
    GRAPHRAG_INDEX_TYPES.append("kso_rag_ui.index.file.graph.GraphRAGIndex")
if USE_NANO_GRAPHRAG:
    GRAPHRAG_INDEX_TYPES.append("kso_rag_ui.index.file.graph.NanoGraphRAGIndex")
if USE_LIGHTRAG:
    GRAPHRAG_INDEX_TYPES.append("kso_rag_ui.index.file.graph.LightRAGIndex")

KSO_RAG_INDEX_TYPES = [
    "kso_rag_ui.index.file.FileIndex",
    *GRAPHRAG_INDEX_TYPES,
]

GRAPHRAG_INDICES = [
    {
        "name": graph_type.split(".")[-1].replace("Index", "")
        + " Collection",  # get last name
        "config": {
            "supported_file_types": (
                ".png, .jpeg, .jpg, .tiff, .tif, .pdf, .xls, .xlsx, .doc, .docx, "
                ".pptx, .csv, .html, .mhtml, .txt, .md, .zip"
            ),
            "private": True,
        },
        "index_type": graph_type,
    }
    for graph_type in GRAPHRAG_INDEX_TYPES
]

KSO_RAG_INDICES = [
    {
        "name": "File Collection",
        "config": {
            "supported_file_types": (
                ".png, .jpeg, .jpg, .tiff, .tif, .pdf, .xls, .xlsx, .doc, .docx, "
                ".pptx, .csv, .html, .mhtml, .txt, .md, .zip"
            ),
            "private": True,
        },
        "index_type": "kso_rag_ui.index.file.index.FileIndex",
    },
    *GRAPHRAG_INDICES,
]
