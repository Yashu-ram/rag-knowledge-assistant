import os
import json
import logging
import sys
from pathlib import Path
from datetime import datetime
from typing import Any

import requests
import streamlit as st

# Streamlit runs this file directly rather than as a Python package.
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from app.ui.api_client import request_backend, indexed_document_ids

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

API_BASE_URL = os.getenv("API_BASE_URL", "http://localhost:8000")
SUPPORTED_FILE_TYPES = ["pdf", "txt", "text", "md", "markdown", "docx", "csv", "json"]

st.set_page_config(
    page_title="HireKiwi RAG Assistant",
    page_icon=":material/chat:",
    layout="wide",
)


def inject_styles() -> None:
    st.html(
        """
        <style>
        .stApp {
            background:
                radial-gradient(circle at 54% 16%, rgba(96, 102, 255, 0.16), transparent 22%),
                radial-gradient(circle at 70% 64%, rgba(58, 224, 190, 0.13), transparent 25%),
                linear-gradient(180deg, #fbfbff 0%, #f5f6fb 100%);
        }
        [data-testid="stSidebar"] {
            background:
                radial-gradient(circle at 25% 5%, rgba(112, 104, 255, 0.28), transparent 30%),
                linear-gradient(180deg, #11131c 0%, #171a27 50%, #0d1018 100%);
            border-right: 1px solid rgba(255, 255, 255, 0.08);
        }
        [data-testid="stSidebar"] * { color: #f8fafc; }
        [data-testid="stSidebar"] .stCaption,
        [data-testid="stSidebar"] label,
        [data-testid="stSidebar"] p {
            color: rgba(248, 250, 252, .74) !important;
        }
        [data-testid="stSidebar"] button {
            border-radius: 14px !important;
            border: 1px solid rgba(255,255,255,.12) !important;
            background: rgba(255,255,255,.08) !important;
            color: #fff !important;
        }
        [data-testid="stSidebar"] button:hover {
            background: rgba(112,104,255,.28) !important;
            border-color: rgba(154,148,255,.6) !important;
        }
        [data-testid="stSidebar"] [data-baseweb="select"] > div,
        [data-testid="stSidebar"] [data-baseweb="popover"] {
            background: #0b0e14 !important;
            border-color: rgba(255,255,255,.14) !important;
            border-radius: 14px !important;
        }
        [data-testid="stSidebar"] [data-testid="stFileUploader"] section {
            background: #0b0e14;
            border: 1px solid rgba(255,255,255,.16);
            border-radius: 16px;
            padding: 12px;
        }
        [data-testid="stSidebar"] [data-testid="stExpander"] {
            background: rgba(255,255,255,.05);
            border-radius: 16px;
            border: 1px solid rgba(255,255,255,.08);
        }
        section.main > div { padding-top: 2rem; }
        .hk-auth-shell {
            max-width: 1060px;
            margin: 5vh auto 0;
            padding: 18px;
            border-radius: 0;
            background:
                radial-gradient(circle at 92% 8%, rgba(149, 97, 255, .14), transparent 24%);
            box-shadow: none;
            border: 0;
        }
        .hk-auth-panel {
            padding: 22px 28px;
            border-radius: 0;
            background: transparent;
            border: 0;
            box-shadow: none;
        }
        .hk-auth-hero {
            min-height: 560px;
            border-radius: 56px;
            padding: 54px;
            background:
                radial-gradient(circle at 78% 35%, rgba(255,255,255,.36), transparent 18%),
                linear-gradient(145deg, #4965ff, #6578ff);
            color: white;
            display: flex;
            flex-direction: column;
            justify-content: space-between;
            overflow: hidden;
        }
        .hk-auth-hero h1 {
            font-size: 40px;
            line-height: 1.16;
            margin: 0 0 18px;
            letter-spacing: 0;
        }
        .hk-auth-hero p {
            font-size: 16px;
            opacity: .9;
            max-width: 390px;
        }
        .hk-illustration {
            height: 210px;
            border-radius: 44px;
            background:
                radial-gradient(circle at 50% 22%, #fff 0 9%, transparent 10%),
                linear-gradient(180deg, rgba(255,255,255,.18), rgba(255,255,255,.04));
            border: 1px solid rgba(255,255,255,.24);
            display: flex;
            align-items: end;
            justify-content: center;
            padding-bottom: 36px;
            font-weight: 700;
            font-size: 22px;
        }
        .hk-brand {
            display: flex;
            align-items: center;
            justify-content: center;
            gap: 16px;
            font-weight: 900;
            color: #5267f6;
            font-size: 38px;
            text-align: center;
            margin-bottom: 28px;
            letter-spacing: 0;
        }
        .hk-brand-mark {
            width: 58px;
            height: 58px;
            border-radius: 18px;
            background:
                radial-gradient(circle at 36% 30%, #ffffff 0 18%, transparent 20%),
                linear-gradient(145deg, #5668ff, #8d67ff);
            box-shadow: 0 8px 22px rgba(86,104,255,.25);
        }
        .hk-auth-title {
            text-align: center;
            font-size: 34px;
            font-weight: 800;
            margin: 0;
            color: #0f172a !important;
        }
        .hk-muted {
            color: #7c8494 !important;
            text-align: center;
        }
        .hk-auth-panel label,
        .hk-auth-panel p,
        .hk-auth-panel span,
        div[data-testid="stTextInput"] label {
            color: #475569 !important;
        }
        div[data-testid="stTextInput"] > div > div {
            border-radius: 999px !important;
            background: #f4f6fb !important;
            border: 1px solid #dfe5f0 !important;
            min-height: 64px !important;
            box-shadow: 0 10px 24px rgba(15, 23, 42, .06);
        }
        div[data-testid="stTextInput"] input {
            background: transparent !important;
            color: #111827 !important;
            border-radius: 999px !important;
            min-height: 64px !important;
            padding-left: 20px !important;
            padding-right: 20px !important;
            font-size: 16px !important;
        }
        div[data-testid="stTextInput"] input::placeholder {
            color: #8a94a6 !important;
        }
        .stButton button,
        div[data-testid="stFormSubmitButton"] button {
            border-radius: 999px !important;
            min-height: 58px;
            font-weight: 700 !important;
            box-shadow: 0 14px 30px rgba(15, 23, 42, .10);
        }
        div[data-testid="stFormSubmitButton"] button[kind="primary"],
        .stButton button[kind="primary"] {
            background: linear-gradient(135deg, #ff4b55, #ff646d) !important;
            border: 0 !important;
        }
        .hk-chat-title {
            max-width: 960px;
            margin-left: auto;
            margin-right: auto;
        }
        .hk-welcome {
            max-width: 960px;
            margin: 8vh auto 28px;
            text-align: center;
        }
        .hk-orb {
            width: 74px;
            height: 74px;
            margin: 0 auto 18px;
            border-radius: 999px;
            background:
                radial-gradient(circle at 34% 30%, #ffffff, rgba(255,255,255,.1) 24%),
                radial-gradient(circle at 68% 70%, #49e7d3, transparent 28%),
                linear-gradient(145deg, #7368ff, #ad77ff);
            box-shadow: 0 18px 48px rgba(96,102,255,.38), 0 0 0 14px rgba(112,104,255,.08);
        }
        .hk-welcome h1 {
            margin: 0;
            font-size: 34px;
            line-height: 1.22;
            letter-spacing: 0;
            color: #111827 !important;
        }
        .hk-welcome span {
            color: #5f63ff !important;
        }
        .hk-sidebar-profile {
            border: 1px solid rgba(255,255,255,.12);
            border-radius: 22px;
            padding: 14px;
            background: rgba(255,255,255,.08);
            margin-top: 12px;
        }
        .hk-sidebar-user {
            display: flex;
            gap: 10px;
            align-items: center;
            padding: 12px;
            border-radius: 18px;
            background: rgba(255,255,255,.08);
            border: 1px solid rgba(255,255,255,.10);
            margin-top: 24px;
        }
        .hk-user-avatar {
            width: 34px;
            height: 34px;
            border-radius: 999px;
            background: linear-gradient(135deg, #5f63ff, #9b7cff);
            display: grid;
            place-items: center;
            font-weight: 800;
            color: #fff;
        }
        .hk-small {
            color: #7d7d86;
            font-size: 13px;
        }
        .hk-topbar {
            max-width: 960px;
            margin: 0 auto;
        }
        .hk-chat-title {
            margin-top: 24px;
            margin-bottom: 12px;
        }
        .hk-chat-title h2 {
            margin: 0;
            color: #111827;
        }
        [data-testid="stChatMessage"] {
            background: #ffffff !important;
            color: #111827 !important;
            border: 1px solid #e2e5ee;
        }
        [data-testid="stChatMessage"] [data-testid="stMarkdownContainer"],
        [data-testid="stChatMessage"] [data-testid="stMarkdownContainer"] p,
        [data-testid="stChatMessage"] [data-testid="stMarkdownContainer"] li,
        [data-testid="stChatMessage"] [data-testid="stMarkdownContainer"] h1,
        [data-testid="stChatMessage"] [data-testid="stMarkdownContainer"] h2,
        [data-testid="stChatMessage"] [data-testid="stMarkdownContainer"] h3,
        [data-testid="stChatMessage"] [data-testid="stMarkdownContainer"] strong,
        [data-testid="stChatMessage"] [data-testid="stMarkdownContainer"] td,
        [data-testid="stChatMessage"] [data-testid="stMarkdownContainer"] th,
        [data-testid="stChatMessage"] [data-testid="stExpander"] summary {
            color: #111827 !important;
        }
        [data-testid="stChatMessage"] [data-testid="stCaptionContainer"] p {
            color: #526074 !important;
        }
        [data-testid="stChatMessage"] a { color: #245ac6 !important; }
        [data-testid="stChatMessage"] code {
            color: #1f2937 !important;
            background: #eef1f6 !important;
        }
        .hk-nav-chip {
            padding: 12px 14px;
            border-radius: 16px;
            background: rgba(255,255,255,.06);
            border: 1px solid rgba(255,255,255,.08);
            margin: 6px 0;
        }
        .hk-history-group {
            margin-top: 16px;
            margin-bottom: 6px;
            color: rgba(248,250,252,.62);
            font-size: 13px;
            font-weight: 700;
        }
        div[data-testid="stChatInput"] {
            max-width: 900px;
            margin: 34px auto 0;
        }
        div[data-testid="stChatInput"] > div {
            min-height: 158px !important;
            border-radius: 28px !important;
            border: 2px solid rgba(129, 112, 255, .46) !important;
            background:
                radial-gradient(circle at 98% 100%, rgba(255, 206, 93, .18), transparent 22%),
                rgba(255,255,255,.78) !important;
            box-shadow: 0 24px 70px rgba(87, 91, 255, .16) !important;
            align-items: flex-end !important;
            padding: 14px 16px !important;
        }
        div[data-testid="stChatInput"] textarea {
            min-height: 116px !important;
            color: #151827 !important;
            background: transparent !important;
            font-size: 15px !important;
            padding-top: 18px !important;
        }
        div[data-testid="stChatInput"] textarea::placeholder {
            color: #7f8798 !important;
        }
        div[data-testid="stChatInput"] button {
            border-radius: 999px !important;
            width: 42px !important;
            height: 42px !important;
            min-height: 42px !important;
            opacity: 1 !important;
            visibility: visible !important;
            background: #ffffff !important;
            color: #151827 !important;
            border: 1px solid rgba(95, 99, 255, .22) !important;
            box-shadow: 0 8px 22px rgba(17, 24, 39, .10) !important;
        }
        div[data-testid="stChatInput"] button svg,
        div[data-testid="stChatInput"] button span {
            color: currentColor !important;
            opacity: 1 !important;
        }
        div[data-testid="stChatInput"] button[kind="primary"] {
            background: linear-gradient(135deg, #6b5cff, #946cff) !important;
            border: 0 !important;
            color: #ffffff !important;
            box-shadow: 0 12px 26px rgba(95, 99, 255, .36) !important;
        }
        div[data-testid="stChatInput"] button:disabled,
        div[data-testid="stChatInput"] button[disabled] {
            opacity: 1 !important;
            filter: none !important;
        }
        div[data-testid="stChatInput"] button[kind="primary"]:disabled,
        div[data-testid="stChatInput"] button[kind="primary"][disabled],
        div[data-testid="stChatInput"] button:last-of-type {
            background: linear-gradient(135deg, #5f63ff, #7a5cff) !important;
            color: #ffffff !important;
            border: 0 !important;
            box-shadow: 0 14px 28px rgba(95, 99, 255, .38) !important;
        }
        div[data-testid="stChatInput"] label,
        div[data-testid="stChatInput"] [role="button"] {
            opacity: 1 !important;
            visibility: visible !important;
            color: #151827 !important;
            border-radius: 999px !important;
        }
        div[data-testid="stVerticalBlockBorderWrapper"]:has(#hk-composer-anchor) {
            max-width: 760px;
            margin: 22px auto 0;
            padding: 14px !important;
            border-radius: 28px !important;
            border: 2px solid rgba(129, 112, 255, .46) !important;
            background:
                radial-gradient(circle at 98% 100%, rgba(255, 206, 93, .18), transparent 22%),
                rgba(255,255,255,.72) !important;
            box-shadow: 0 24px 70px rgba(87, 91, 255, .16) !important;
        }
        #hk-composer-anchor {
            display: none;
        }
        div[data-testid="stTextArea"] textarea {
            min-height: 118px !important;
            border-radius: 22px 22px 12px 12px !important;
            border: 0 !important;
            background: transparent !important;
            color: #151827 !important;
            font-size: 15px !important;
            box-shadow: none !important;
        }
        div[data-testid="stTextArea"] textarea::placeholder {
            color: #8b94a7 !important;
        }
        div[data-testid="stFileUploader"] section {
            min-height: 42px !important;
            padding: 7px 12px !important;
            border-radius: 999px !important;
            border: 1px solid rgba(95,99,255,.20) !important;
            background: rgba(95,99,255,.06) !important;
        }
        div[data-testid="stFileUploader"] section [data-testid="stFileUploaderDropzoneInstructions"] {
            display: none !important;
        }
        div[data-testid="stFileUploader"] section button {
            border-radius: 999px !important;
            min-height: 34px !important;
            background: #ffffff !important;
            color: #5f63ff !important;
            border: 1px solid rgba(95,99,255,.22) !important;
            box-shadow: none !important;
        }
        div[data-testid="stFileUploader"] small {
            display: none !important;
        }
        div[data-testid="stFormSubmitButton"] button {
            border-radius: 999px !important;
        }
        </style>
        """
    )


def init_state() -> None:
    st.session_state.setdefault("token", None)
    st.session_state.setdefault("user", None)
    st.session_state.setdefault("auth_view", "login")
    st.session_state.setdefault("conversation_id", None)
    st.session_state.setdefault("messages", [])
    st.session_state.setdefault("selected_document_ids", [])
    st.session_state.setdefault("chat_documents", {})
    st.session_state.setdefault("conversation_documents", {})
    st.session_state.setdefault("last_signup_email", "")
    st.session_state.setdefault("signup_success", False)
    st.session_state.setdefault("documents", [])
    st.session_state.setdefault("documents_error", None)
    st.session_state.setdefault("document_selection_dirty", False)
    st.session_state.setdefault("upload_errors", [])


def api(method: str, path: str, **kwargs) -> Any:
    uploading = path == "/documents/upload"
    on_progress = kwargs.pop("on_progress", None)
    timeout = kwargs.pop("timeout", 300 if uploading else 120)
    if uploading and on_progress:
        kwargs["params"] = {**kwargs.get("params", {}), "stream": "true"}
        kwargs["stream"] = True
    headers = kwargs.pop("headers", {})
    token = st.session_state.get("token")
    if token:
        headers["Authorization"] = f"Bearer {token}"
    try:
        response = request_backend(method, API_BASE_URL, path, token=token,
                                   headers=headers, timeout=timeout, **kwargs)
    except requests.Timeout as exc:
        if not uploading:
            raise
        raise RuntimeError(f"Document upload timed out after {timeout} seconds. Check document status before retrying.") from exc
    if response.status_code == 401:
        logger.warning("[STREAMLIT-AUTH] session expired")
        response.close()
        st.session_state["ingestion_in_progress"] = False
        st.session_state["token"] = None
        st.session_state["user"] = None
        st.error("Session expired. Please log in again.")
        st.rerun()
    try:
        if response.status_code >= 400:
            try:
                message = response.json().get("detail") or response.json().get("error", {}).get("message")
            except ValueError:
                message = None
            if message:
                raise RuntimeError(str(message))
            response.raise_for_status()
        if uploading and on_progress:
            for line in response.iter_lines(chunk_size=1):
                if not line:
                    continue
                event = json.loads(line)
                if event["type"] == "progress":
                    on_progress(event["stage"])
                elif event["type"] == "error":
                    raise RuntimeError(event["message"])
                elif event["type"] == "result":
                    return event["document"]
            raise RuntimeError("Upload ended before ingestion completed. Check document status before retrying.")
        return response.json() if response.content else None
    except (requests.Timeout, requests.ConnectionError) as exc:
        if uploading:
            raise RuntimeError("Upload connection timed out or was interrupted. Ingestion may still be processing; check document status before retrying.") from exc
        raise
    finally:
        response.close()


def safe_api(method: str, path: str, default: Any) -> Any:
    try:
        return api(method, path)
    except Exception:
        return default


def post_auth(path: str, payload: dict) -> tuple[bool, dict | str]:
    try:
        response = request_backend("POST", API_BASE_URL, path, json=payload, timeout=30)
        logger.info("[STREAMLIT-AUTH] action=%s status=%s", path, response.status_code)
        if response.status_code >= 400:
            try:
                return False, response.json().get("detail", "Request failed")
            except Exception:
                return False, response.text or "Request failed"
        return True, response.json()
    except Exception as exc:
        return False, str(exc)


def auth_screen() -> None:
    st.markdown('<div class="hk-auth-shell">', unsafe_allow_html=True)
    left, right = st.columns([1.08, 1], gap="large", vertical_alignment="center")
    with left:
        st.markdown(
            """
            <div class="hk-auth-hero">
                <div>
                    <h1>Chat with your documents, grounded in evidence</h1>
                    <p>Upload PDFs, text, and Markdown files. Ask questions and get answers with exact source excerpts.</p>
                </div>
                <div class="hk-illustration">HireKiwi RAG Assistant</div>
            </div>
            """,
            unsafe_allow_html=True,
        )
    with right:
        st.markdown('<div class="hk-auth-panel">', unsafe_allow_html=True)
        st.markdown(
            """
            <div class="hk-brand">
                <div class="hk-brand-mark"></div>
                <div>HireKiwi AI</div>
            </div>
            """,
            unsafe_allow_html=True,
        )
        if st.session_state.auth_view == "login":
            login_form()
        else:
            signup_form()
        st.markdown("</div>", unsafe_allow_html=True)
    st.markdown("</div>", unsafe_allow_html=True)


def login_form() -> None:
    st.markdown('<h1 class="hk-auth-title">Welcome back</h1>', unsafe_allow_html=True)
    st.markdown('<p class="hk-muted">Log in to continue to your knowledge assistant.</p>', unsafe_allow_html=True)
    if st.session_state.get("signup_success"):
        st.success("Account created. Please log in.")
        st.session_state["signup_success"] = False
    with st.form("login_form", border=False):
        email = st.text_input("Email", value=st.session_state.get("last_signup_email", ""), placeholder="Email")
        password = st.text_input("Password", type="password", placeholder="Password")
        submitted = st.form_submit_button("Log in", type="primary", width="stretch")
    if submitted:
        ok, result = post_auth("/auth/login", {"email": email, "password": password})
        if ok:
            st.session_state["token"] = result["access_token"]
            st.session_state["user"] = result
            st.session_state["messages"] = []
            st.session_state["conversation_id"] = None
            st.session_state["chat_documents"] = {}
            st.session_state["selected_document_ids"] = []
            st.session_state["document_selection_dirty"] = True
            st.rerun()
        else:
            st.error(f"Login failed: {result}")

    st.caption("New here?")
    if st.button("Create an account", width="stretch"):
        st.session_state.auth_view = "signup"
        st.rerun()


def signup_form() -> None:
    st.markdown('<h1 class="hk-auth-title">Create account</h1>', unsafe_allow_html=True)
    st.markdown('<p class="hk-muted">Use any password characters you want. Just confirm it correctly.</p>', unsafe_allow_html=True)
    with st.form("signup_form", border=False):
        full_name = st.text_input("Full name", placeholder="Full name")
        email = st.text_input("Email", placeholder="Email")
        password = st.text_input("Password", type="password", placeholder="Password")
        confirm_password = st.text_input("Confirm password", type="password", placeholder="Confirm password")
        submitted = st.form_submit_button("Create account", type="primary", width="stretch")
    if submitted:
        if password != confirm_password:
            st.error("Passwords do not match.")
            return
        ok, result = post_auth("/auth/signup", {"full_name": full_name, "email": email, "password": password})
        if ok:
            st.session_state["last_signup_email"] = email
            st.session_state.auth_view = "login"
            st.session_state["signup_success"] = True
            st.rerun()
        else:
            st.error(f"Signup failed: {result}")

    st.caption("Already have an account?")
    if st.button("Log in", width="stretch"):
        st.session_state.auth_view = "login"
        st.rerun()


def upload_files(files) -> list[str]:
    uploaded_names = []
    st.session_state["upload_failed"] = False
    st.session_state["ingestion_in_progress"] = bool(files)
    if files:
        st.session_state["upload_errors"] = []
    for file in files or []:
        status = st.status(f"Uploading file: {file.name}", expanded=True)
        def progress(stage):
            status.update(label=f"{stage}: {file.name}")
            status.write(stage)
        try:
            response = api(
                "POST",
                "/documents/upload",
                files={"file": (file.name, file.getvalue())},
                on_progress=progress,
            )
            if response.get("status") != "indexed":
                raise RuntimeError("Document ingestion has not completed successfully.")
            status.update(label=f"Indexed: {response['filename']}", state="complete", expanded=False)
            uploaded_names.append(response["filename"])
            warnings = response.get("analysis", {}).get("warnings", [])
            if warnings:
                st.warning(f"{response['filename']}: {len(warnings)} extraction notice(s). Some content needs review.")
                with st.expander("Extraction details"):
                    for warning in warnings:
                        page = f"Page {warning['page']}: " if warning.get("page") else ""
                        st.write(f"{page}{warning['message']}")
            st.session_state["chat_documents"][response["document_id"]] = response["filename"]
            selected = st.session_state["selected_document_ids"]
            if response["document_id"] not in selected:
                selected.append(response["document_id"])
            st.session_state["document_selection_dirty"] = True
            logger.info("[STREAMLIT-DOCS] upload indexed document_id=%s", response["document_id"])
        except Exception as exc:
            st.session_state["upload_failed"] = True
            status.update(label=f"Upload failed: {file.name}", state="error", expanded=True)
            message = f"Upload failed for {getattr(file, 'name', 'file')}: {exc}"
            st.session_state["upload_errors"].append(message)
            st.error(message)
    st.session_state["ingestion_in_progress"] = False
    if files:
        refresh_documents()
    return uploaded_names


def refresh_documents() -> bool:
    """Reload authenticated metadata, never confuse an API error with an empty list."""
    try:
        documents = api("GET", "/documents", timeout=15)
        st.session_state["documents"] = documents
        st.session_state["documents_error"] = None
        valid = indexed_document_ids(documents)
        st.session_state["selected_document_ids"] = [
            doc_id for doc_id in st.session_state["selected_document_ids"] if doc_id in valid
        ]
        st.session_state["chat_documents"] = {
            doc_id: name for doc_id, name in st.session_state["chat_documents"].items() if doc_id in valid
        }
        logger.info("[STREAMLIT-DOCS] list count=%d indexed=%d", len(documents), len(valid))
        return True
    except Exception as exc:
        st.session_state["documents_error"] = str(exc)
        logger.warning("[STREAMLIT-DOCS] list failed error_type=%s", type(exc).__name__)
        return False


def render_documents() -> list[str]:
    st.subheader("Documents")
    st.button("Refresh Documents", icon=":material/refresh:", key="refresh_documents")
    # Fetch on every rerun, including login, refresh, and external sample uploads.
    available = refresh_documents()
    if not available:
        st.error(st.session_state["documents_error"])
    documents = st.session_state["documents"]
    if available and not documents:
        st.info("No documents uploaded for this account yet.")
    for message in st.session_state.get("upload_errors", []):
        st.error(message)
    uploads = st.file_uploader("Upload documents", type=SUPPORTED_FILE_TYPES,
                               accept_multiple_files=True, key="document_uploads")
    if st.button("Upload selected files", icon=":material/upload:", disabled=not uploads):
        upload_files(uploads)
        st.rerun()
    for item in documents:
        status = "completed / indexed" if item["status"] == "indexed" else item["status"]
        sample_label = " (shared sample)" if item.get("is_sample") else ""
        with st.expander(f"{item['filename']}{sample_label} - {status}"):
            st.caption(f"Type: {item['file_type']} | Created: {item['created_at']}")
            st.code(item["document_id"], language=None)
            st.caption(f"Chunks: {item.get('analysis', {}).get('chunk_count', 0)}")
            if item.get("error_message"):
                st.error(item["error_message"])
    options = [item["document_id"] for item in documents if item["status"] == "indexed"] if available else []
    labels = {item["document_id"]: f"{item['filename']}{' [sample]' if item.get('is_sample') else ''} ({item['document_id'][:8]})" for item in documents}
    if st.session_state.pop("document_selection_dirty", False) or "document_selector" not in st.session_state:
        st.session_state["document_selector"] = [
            doc_id for doc_id in st.session_state["selected_document_ids"] if doc_id in options
        ]
    else:
        st.session_state["document_selector"] = [
            doc_id for doc_id in st.session_state["document_selector"] if doc_id in options
        ]
    selected = st.multiselect("Select documents to query", options, format_func=labels.get,
                              key="document_selector", disabled=not options)
    st.session_state["selected_document_ids"] = list(selected)
    st.session_state["chat_documents"] = {
        item["document_id"]: item["filename"] for item in documents if item["document_id"] in selected
    }
    if not selected:
        st.warning("Select an indexed document or upload a file before asking a question.")
    return list(selected)


def load_conversation(conversation_id: str) -> None:
    data = api("GET", f"/conversations/{conversation_id}")
    st.session_state["conversation_id"] = conversation_id
    st.session_state["messages"] = data.get("messages", [])
    st.session_state["chat_documents"] = dict(
        st.session_state["conversation_documents"].get(conversation_id, {})
    )
    if not st.session_state["chat_documents"]:
        for message in data.get("messages", []):
            for citation in message.get("citations") or []:
                st.session_state["chat_documents"][citation["document_id"]] = citation["source_filename"]
    st.session_state["selected_document_ids"] = list(st.session_state["chat_documents"])
    st.session_state["document_selection_dirty"] = True


def greeting() -> str:
    hour = datetime.now().hour
    if hour < 12:
        return "Good morning"
    if hour < 17:
        return "Good afternoon"
    return "Good evening"


def first_name() -> str:
    user = st.session_state.get("user") or {}
    label = user.get("full_name") or user.get("email") or "there"
    return label.split()[0]


def render_sidebar() -> tuple[str | None, str, list[str] | None]:
    user = st.session_state.get("user") or {}
    with st.sidebar:
        st.title("HireKiwi AI", icon=":material/chat:")
        st.caption("AI document assistant")
        selected_docs = render_documents()

        search = st.text_input("Search", placeholder="Search", label_visibility="collapsed")

        st.button("Home", icon=":material/home:", width="stretch", disabled=True)
        st.button("Explore", icon=":material/explore:", width="stretch", disabled=True)
        st.button("Library", icon=":material/folder:", width="stretch", disabled=True)
        st.button("History", icon=":material/history:", width="stretch", disabled=True)

        conversations = safe_api("GET", "/conversations", default=[])
        if search:
            conversations = [
                item for item in conversations if search.lower() in item.get("title", "").lower()
            ]
        if conversations:
            grouped: dict[str, list[dict]] = {"Today": [], "Previous 7 days": [], "Older": []}
            today = datetime.now().date()
            for item in conversations[:15]:
                raw_date = item.get("updated_at") or item.get("created_at") or ""
                try:
                    item_date = datetime.fromisoformat(raw_date.replace("Z", "+00:00")).date()
                    age_days = (today - item_date).days
                except Exception:
                    age_days = 999
                if age_days <= 0:
                    grouped["Today"].append(item)
                elif age_days <= 7:
                    grouped["Previous 7 days"].append(item)
                else:
                    grouped["Older"].append(item)

            for group_name, group_items in grouped.items():
                if not group_items:
                    continue
                st.markdown(f'<div class="hk-history-group">{group_name}</div>', unsafe_allow_html=True)
                for item in group_items:
                    active = item["conversation_id"] == st.session_state.get("conversation_id")
                    label = f"{':material/check_circle:' if active else ':material/chat_bubble:'} {item['title']}"
                    if st.button(label, key=f"conv-{item['conversation_id']}", width="stretch"):
                        load_conversation(item["conversation_id"])
                        st.rerun()
        else:
            st.markdown('<div class="hk-history-group">History</div>', unsafe_allow_html=True)
            st.caption("No chat history yet.")

        st.caption("Microphone")
        mic_col, model_col = st.columns([0.22, 0.78], vertical_alignment="center")
        mic_col.button("", icon=":material/mic:", disabled=True, width="stretch")
        models = safe_api("GET", "/models", default={"available": False, "models": []})
        model_options = ["qwen2.5:1.5b", "qwen2.5:3b"]
        model_name = model_col.selectbox(
            "Change model",
            options=model_options,
            format_func=lambda name: {"qwen2.5:1.5b": "Qwen 2.5 1.5B", "qwen2.5:3b": "Qwen 2.5 3B"}[name],
            label_visibility="collapsed",
        )
        response_mode = st.segmented_control("Reasoning", ["quick", "deep"], default="quick")
        if not models.get("available"):
            st.caption("Local model service is unavailable.")
        elif model_name not in models.get("installed_models", []):
            st.warning(f"Model missing. Run: ollama pull {model_name}")

        initials = (user.get("full_name") or user.get("email") or "User")[:1].upper()
        st.markdown(
            f"""
            <div class="hk-sidebar-user">
                <div class="hk-user-avatar">{initials}</div>
                <div>
                    <strong>{user.get('full_name') or 'User'}</strong><br>
                    <span class="hk-small">{user.get('email', '')}</span>
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )
        if st.button("Log out", icon=":material/logout:", width="stretch"):
            try:
                api("POST", "/auth/logout")
            except Exception:
                pass
            st.session_state.clear()
            st.session_state["auth_view"] = "login"
            st.rerun()

    return model_name or None, response_mode or "quick", selected_docs


def render_top_bar() -> None:
    st.markdown('<div class="hk-topbar">', unsafe_allow_html=True)
    with st.container(horizontal=True, horizontal_alignment="right"):
        if st.button("New chat", icon=":material/add:", type="primary"):
            st.session_state["conversation_id"] = None
            st.session_state["messages"] = []
            st.session_state["chat_documents"] = {}
            st.session_state["selected_document_ids"] = []
            st.session_state["document_selection_dirty"] = True
            st.rerun()
    st.markdown("</div>", unsafe_allow_html=True)


def render_chat_header() -> None:
    if not st.session_state.get("messages"):
        st.markdown(
            f"""
            <div class="hk-welcome">
                <div class="hk-orb"></div>
                <h1>{greeting()}, {first_name()}<br>How can I <span>assist you today?</span></h1>
            </div>
            """,
            unsafe_allow_html=True,
        )
    else:
        st.markdown(
            """
            <div class="hk-chat-title">
                <h2>HireKiwi AI</h2>
            </div>
            """,
            unsafe_allow_html=True,
        )


def render_messages() -> None:
    for message in st.session_state.get("messages", []):
        role = message.get("role", "assistant")
        avatar = ":material/person:" if role == "user" else ":material/smart_toy:"
        with st.chat_message(role, avatar=avatar):
            st.write(message.get("content", ""))
            for citation in message.get("citations") or []:
                if citation.get("anchor"):
                    st.html(f'<span id="{citation["anchor"]}"></span>')
                title = f"[{citation.get('citation_id') or ''}] {citation['source_filename']}"
                with st.expander(title, icon=":material/article:"):
                    if citation.get("page_number"):
                        st.caption(f"Page {citation['page_number']}")
                    if citation.get("section"):
                        st.caption(f"Section: {citation['section']}")
                    st.caption(f"Chunk: {citation['chunk_id']}")
                    st.write(citation["excerpt"])
                    if citation.get("context_excerpt") and citation["context_excerpt"] != citation["excerpt"]:
                        st.caption("Source context")
                        st.write(citation["context_excerpt"])


def submit_question(question: str, files, model_name: str | None, response_mode: str, selected_docs: list[str] | None) -> None:
    uploaded_names = upload_files(files)
    if st.session_state.get("upload_failed") or st.session_state.get("ingestion_in_progress"):
        st.warning("No question was sent. Wait for document ingestion to complete successfully.")
        if files:
            st.rerun()
        return
    if uploaded_names:
        st.toast(f"Indexed {len(uploaded_names)} attached file(s).")
    selected_docs = list(st.session_state["selected_document_ids"])

    if not question.strip():
        if uploaded_names:
            st.rerun()
        return

    if st.session_state.get("documents_error"):
        st.error(st.session_state["documents_error"])
        return
    if not selected_docs:
        st.warning("Select an indexed document before asking a question.")
        return
    logger.info("[RAG-QUERY] selected_documents=%d", len(selected_docs))

    with st.chat_message("user", avatar=":material/person:"):
        st.write(question)

    payload = {
        "question": question,
        "conversation_id": st.session_state.get("conversation_id"),
        "model_name": model_name,
        "response_mode": response_mode,
        "document_ids": selected_docs,
    }
    with st.chat_message("assistant", avatar=":material/smart_toy:"):
        try:
            with st.spinner("Searching documents and generating an answer..."):
                result = api("POST", "/chat/query", json=payload)
        except Exception as exc:
            message = f"The answer request failed: {exc}. Your uploaded documents are still attached; retry your question."
            st.error(message)
            if uploaded_names:
                st.session_state["query_error"] = message
                st.rerun()
            return
        st.session_state["conversation_id"] = result["conversation_id"]
        st.session_state["conversation_documents"][result["conversation_id"]] = dict(
            st.session_state["chat_documents"]
        )
        st.write(result["answer"])
        st.caption(f"{result['answerable']} · model: {result.get('model_used') or 'not invoked'} · {result['latency_ms']} ms")
        for citation in result["citations"]:
            if citation.get("anchor"):
                st.html(f'<span id="{citation["anchor"]}"></span>')
            with st.expander(f"[{citation.get('citation_id') or ''}] {citation['source_filename']}", expanded=True, icon=":material/article:"):
                if citation.get("page_number"):
                    st.caption(f"Page {citation['page_number']}")
                if citation.get("section"):
                    st.caption(f"Section: {citation['section']}")
                st.caption(f"Chunk: {citation['chunk_id']}")
                st.write(citation["excerpt"])
                if citation.get("context_excerpt") and citation["context_excerpt"] != citation["excerpt"]:
                    st.caption("Source context")
                    st.write(citation["context_excerpt"])

    st.session_state["messages"].append({"role": "user", "content": question})
    st.session_state["messages"].append(
        {
            "role": "assistant",
            "content": result["answer"],
            "citations": result["citations"],
            "model_used": result.get("model_used"),
            "answerable": result["answerable"],
        }
    )
    st.rerun()


def handle_chat_submission(model_name: str | None, response_mode: str, selected_docs: list[str] | None) -> None:
    _, composer_col, _ = st.columns([0.08, 0.84, 0.08])
    with composer_col:
        if st.session_state["chat_documents"]:
            st.caption("Attached: " + ", ".join(st.session_state["chat_documents"].values()))
        submission = st.chat_input(
            "Ask anything from your uploaded documents",
            key="composer_chat",
            accept_file="multiple",
            disabled=st.session_state.get("ingestion_in_progress", False),
            file_type=SUPPORTED_FILE_TYPES,
            accept_audio=True,
            submit_mode="disable",
            width="stretch",
        )
    if not submission:
        return

    question = submission.text if hasattr(submission, "text") else str(submission)
    files = submission.files if hasattr(submission, "files") else []
    submit_question(question, files, model_name, response_mode, selected_docs)


def main_app() -> None:
    model_name, response_mode, selected_docs = render_sidebar()
    render_top_bar()
    render_chat_header()
    if st.session_state.get("query_error"):
        st.error(st.session_state.pop("query_error"))
    render_messages()
    handle_chat_submission(model_name, response_mode, selected_docs)


inject_styles()
init_state()

if not st.session_state.get("token"):
    auth_screen()
else:
    main_app()
