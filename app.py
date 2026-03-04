import os

from theflow.settings import settings as flowsettings

KSO_RAG_APP_DATA_DIR = getattr(flowsettings, "KSO_RAG_APP_DATA_DIR", ".")
KSO_RAG_GRADIO_SHARE = getattr(flowsettings, "KSO_RAG_GRADIO_SHARE", False)
GRADIO_TEMP_DIR = os.getenv("GRADIO_TEMP_DIR", None)
# override GRADIO_TEMP_DIR if it's not set
if GRADIO_TEMP_DIR is None:
    GRADIO_TEMP_DIR = os.path.join(KSO_RAG_APP_DATA_DIR, "gradio_tmp")
    os.environ["GRADIO_TEMP_DIR"] = GRADIO_TEMP_DIR

# Read server settings from env (set by launch.sh)
GRADIO_SERVER_NAME = os.getenv("GRADIO_SERVER_NAME", "127.0.0.1")
GRADIO_SERVER_PORT = int(os.getenv("GRADIO_SERVER_PORT", "7860"))
# Only open browser when running locally (not in Docker / headless environments)
INBROWSER = GRADIO_SERVER_NAME == "127.0.0.1"

# Fix gradio_client: handle boolean JSON Schema (e.g. additionalProperties: true)
# so APIInfoParseError "Cannot parse schema True" and "const" in schema TypeError are avoided
try:
    import gradio_client.utils as _gc_utils

    _json_schema_orig = getattr(_gc_utils, "_json_schema_to_python_type", None)
    if _json_schema_orig is not None:

        def _json_schema_patched(schema, defs):
            if isinstance(schema, bool):
                return "Any"
            return _json_schema_orig(schema, defs)

        _gc_utils._json_schema_to_python_type = _json_schema_patched
except Exception:
    pass

from kso_rag_ui.main import App  # noqa

app = App()
demo = app.make()
demo.queue().launch(
    favicon_path=app._favicon,
    server_name=GRADIO_SERVER_NAME,
    server_port=GRADIO_SERVER_PORT,
    inbrowser=INBROWSER,
    allowed_paths=[
        "src/ui/kso_rag_ui/assets",
        GRADIO_TEMP_DIR,
    ],
    share=KSO_RAG_GRADIO_SHARE or (GRADIO_SERVER_NAME == "0.0.0.0"),
)
