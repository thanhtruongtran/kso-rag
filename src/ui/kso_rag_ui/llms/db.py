from typing import Type

from kso_rag_ui.db.engine import engine
from sqlalchemy import JSON, Boolean, Column, String
from sqlalchemy.orm import DeclarativeBase
from theflow.settings import settings as flowsettings
from theflow.utils.modules import import_dotted_string


class Base(DeclarativeBase):
    pass


class BaseLLMTable(Base):
    """Base table to store language model"""

    __abstract__ = True

    name = Column(String, primary_key=True, unique=True)
    spec = Column(JSON, default={})
    default = Column(Boolean, default=False)


_base_llm: Type[BaseLLMTable] = (
    import_dotted_string(flowsettings.KSO_RAG_TABLE_LLM, safe=False)
    if hasattr(flowsettings, "KSO_RAG_TABLE_LLM")
    else BaseLLMTable
)


class LLMTable(_base_llm):  # type: ignore
    __tablename__ = "llm_table"


if not getattr(flowsettings, "KSO_RAG_ENABLE_ALEMBIC", False):
    LLMTable.metadata.create_all(engine)
