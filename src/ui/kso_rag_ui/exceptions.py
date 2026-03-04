class KSO_RAG_Exception(Exception):
    pass


class HookNotDeclared(KSO_RAG_Exception):
    pass


class HookAlreadyDeclared(KSO_RAG_Exception):
    pass
