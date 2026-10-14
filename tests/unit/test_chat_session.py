import ast
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock
from app.ui.api_client import indexed_document_ids


def chat_functions(api):
    path = Path(__file__).resolve().parents[2] / "app/ui/streamlit_app.py"
    tree = ast.parse(path.read_text(encoding="utf-8"))
    names = {"upload_files", "submit_question", "load_conversation", "refresh_documents"}
    tree.body = [node for node in tree.body if isinstance(node, ast.FunctionDef) and node.name in names]
    st = MagicMock()
    st.session_state = {
        "chat_documents": {}, "conversation_documents": {},
        "messages": [], "conversation_id": None,
        "documents": [], "selected_document_ids": [], "upload_errors": [],
    }
    namespace = {"st": st, "api": api, "logger": MagicMock(), "indexed_document_ids": indexed_document_ids}
    exec(compile(tree, str(path), "exec"), namespace)
    return namespace, st


def answer():
    return {
        "conversation_id": "chat-1", "answer": "Document answer",
        "answerable": "answerable", "model_used": "qwen2.5:3b",
        "latency_ms": 10, "citations": [],
    }


def test_followup_reuses_all_uploaded_documents():
    api = MagicMock(side_effect=[
        {"document_id": "doc-1", "filename": "one.txt", "status": "indexed"},
        {"document_id": "doc-2", "filename": "two.txt", "status": "indexed"},
        [{"document_id": "doc-1", "filename": "one.txt", "status": "indexed"},
         {"document_id": "doc-2", "filename": "two.txt", "status": "indexed"}], answer(), answer(),
    ])
    functions, st = chat_functions(api)
    files = [SimpleNamespace(name=name, getvalue=lambda: b"content") for name in ("one.txt", "two.txt")]
    functions["submit_question"]("First question", files, "qwen2.5:3b", "quick", None)
    functions["submit_question"]("Follow-up", [], "qwen2.5:3b", "quick", None)
    assert api.call_count == 5
    assert api.call_args_list[2].args == ("GET", "/documents")
    assert len(st.session_state["documents"]) == 2
    assert api.call_args.kwargs["json"]["document_ids"] == ["doc-1", "doc-2"]
    assert api.call_args.kwargs["json"]["conversation_id"] == "chat-1"
    assert len(st.session_state["messages"]) == 4


def test_failed_answer_keeps_attachment_and_displays_error():
    api = MagicMock(side_effect=RuntimeError("Connection refused"))
    functions, st = chat_functions(api)
    st.session_state["chat_documents"] = {"doc-1": "one.txt"}
    st.session_state["selected_document_ids"] = ["doc-1"]
    functions["submit_question"]("Question", [], None, "quick", None)
    assert st.session_state["chat_documents"] == {"doc-1": "one.txt"}
    assert st.session_state["messages"] == []
    assert "Connection refused" in st.error.call_args.args[0]


def test_history_restores_session_attachments():
    functions, st = chat_functions(MagicMock(return_value={"messages": []}))
    st.session_state["conversation_documents"] = {"chat-1": {"doc-1": "one.txt"}}
    functions["load_conversation"]("chat-1")
    assert st.session_state["chat_documents"] == {"doc-1": "one.txt"}
