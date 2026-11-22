from abc import ABC
from abc import abstractmethod
from typing import Optional
from utils.config_handler import rag_conf


class BaseModelFactory(ABC):
    @abstractmethod
    def generator(self):
        pass


class ChatModelFactory(BaseModelFactory):
    def generator(self):
        # 延迟 import — 避免 import 链上触发 langchain_community 加载 (~74ms)
        from langchain_community.chat_models.tongyi import ChatTongyi
        return ChatTongyi(model=rag_conf["chat_model_name"])


class EmbeddingsFactory(BaseModelFactory):
    def generator(self):
        # 延迟 import — 避免 import 链上触发 langchain_community.embeddings 加载 (~2ms)
        from langchain_community.embeddings import DashScopeEmbeddings
        return DashScopeEmbeddings(model=rag_conf["embedding_model_name"])


_chat_model = None
_embed_model = None


def get_chat_model():
    """懒加载：首次调用时才 import LangChain 社区模型并创建 ChatTongyi 实例，
    避免 import 链上立即触发 Pydantic 验证和 API Key 检查"""
    global _chat_model
    if _chat_model is None:
        _chat_model = ChatModelFactory().generator()
    return _chat_model


def get_embed_model():
    """懒加载：首次调用时才 import LangChain 社区 embedding 并创建 DashScopeEmbeddings 实例"""
    global _embed_model
    if _embed_model is None:
        _embed_model = EmbeddingsFactory().generator()
    return _embed_model
