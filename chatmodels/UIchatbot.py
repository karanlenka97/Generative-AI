import uuid

import streamlit as st
from transformers import pipeline

st.set_page_config(
    page_title="Ava ChatBot",
    page_icon="🤖",
    layout="wide",
)


@st.cache_resource
def load_chatbot():
    return pipeline(
        "text-generation",
        model="TinyLlama/TinyLlama-1.1B-Chat-v1.0",
    )


def create_new_chat():
    chat_id = str(uuid.uuid4())
    st.session_state.chats[chat_id] = {
        "title": "New chat",
        "messages": [],
    }
    st.session_state.active_chat_id = chat_id


def open_chat(chat_id):
    st.session_state.active_chat_id = chat_id


def queue_prompt(prompt):
    st.session_state.pending_prompt = prompt


if "chats" not in st.session_state:
    first_chat_id = str(uuid.uuid4())
    st.session_state.chats = {
        first_chat_id: {
            "title": "New chat",
            "messages": [],
        }
    }
    st.session_state.active_chat_id = first_chat_id


st.markdown(
    """
    <style>
    html,
    body,
    .stApp,
    [data-testid="stAppViewContainer"],
    [data-testid="stMain"],
    [data-testid="stBottom"],
    [data-testid="stBottom"] > div,
    [data-testid="stBottomBlockContainer"] {
        background: #ffffff !important;
        background-image: none !important;
        color: #111827 !important;
        box-shadow: none !important;
    }

    [data-testid="stHeader"],
    [data-testid="stToolbar"],
    [data-testid="stSidebar"] {
        background: #ffffff !important;
        color: #111827 !important;
    }

    [data-testid="stSidebar"] {
        border-right: 1px solid #e5e7eb !important;
    }

    [data-testid="stSidebar"] * {
        color: #111827 !important;
    }

    .block-container {
        max-width: 1050px;
        padding-top: 2.5rem;
        padding-bottom: 2rem;
    }

    .ava-brand {
        display: flex;
        align-items: center;
        gap: 12px;
        margin: 0.5rem 0 1.5rem 0;
        color: #111827;
        font-size: 1.35rem;
        font-weight: 700;
    }

    .ava-brand-icon {
        font-size: 2.4rem;
        line-height: 1;
    }

    .welcome {
        padding: 3rem 1rem 1.5rem 1rem;
        text-align: center;
    }

    .welcome-icon {
        display: flex;
        width: 96px;
        height: 96px;
        align-items: center;
        justify-content: center;
        margin: 0 auto 1.2rem auto;
        border: 1px solid #e5e7eb;
        border-radius: 28px;
        background: #f8fafc;
        font-size: 3.8rem;
        line-height: 1;
    }

    .welcome h1 {
        margin-bottom: 0.5rem;
        color: #111827 !important;
        font-size: 2.2rem;
    }

    .welcome p {
        color: #4b5563 !important;
        font-size: 1rem;
    }

    [data-testid="stChatMessage"] {
        color: #111827 !important;
        background: #ffffff !important;
        border: 1.5px solid #cbd5e1 !important;
        border-radius: 18px !important;
        padding: 18px 22px !important;
        margin: 12px 0 18px 0 !important;
        box-shadow: 0 3px 10px rgba(15, 23, 42, 0.06) !important;
    }

    [data-testid="stChatMessage"] *,
    [data-testid="stChatMessage"] p,
    [data-testid="stChatMessage"] li,
    [data-testid="stChatMessage"] span,
    [data-testid="stChatMessage"] [data-testid="stMarkdownContainer"] {
        color: #111827 !important;
        line-height: 1.65 !important;
    }

    [data-testid="stChatInput"] {
        background: #ffffff !important;
        border: 1.5px solid #cbd5e1 !important;
        border-radius: 16px !important;
        box-shadow: none !important;
    }

    [data-testid="stChatInput"] textarea,
    [data-testid="stChatInput"] textarea:focus {
        color: #111827 !important;
        background: #ffffff !important;
        caret-color: #111827 !important;
    }

    [data-testid="stChatInput"] textarea::placeholder {
        color: #6b7280 !important;
        opacity: 1 !important;
    }

    [data-testid="stChatInputSubmitButton"] {
        color: #111111 !important;
        background: #ffffff !important;
        border: none !important;
        box-shadow: none !important;
    }

    [data-testid="stChatInputSubmitButton"] svg,
    [data-testid="stChatInputSubmitButton"] svg path {
        color: #111111 !important;
        fill: #111111 !important;
        stroke: #111111 !important;
    }

    div.stButton > button {
        min-height: 2.8rem;
        border: 1px solid #d1d5db;
        border-radius: 12px;
        background: #ffffff !important;
        color: #111827 !important;
    }

    div.stButton > button:hover {
        border-color: #8b5cf6 !important;
        background: #faf8ff !important;
        color: #5b21b6 !important;
    }

    [data-testid="stSelectbox"] div,
    [data-testid="stSelectbox"] input {
        color: #111827 !important;
        background-color: #ffffff !important;
    }

    hr {
        border-color: #e5e7eb !important;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

with st.sidebar:
    st.markdown(
        """
        <div class="ava-brand">
            <span class="ava-brand-icon">🤖</span>
            <span>Ava ChatBot</span>
        </div>
        """,
        unsafe_allow_html=True,
    )

    st.button(
        "＋  New chat",
        use_container_width=True,
        on_click=create_new_chat,
    )

    st.selectbox(
        "Model",
        ["TinyLlama · Local"],
        help="Ava runs TinyLlama on your computer.",
    )

    st.markdown("### History")

    for chat_id, chat in reversed(list(st.session_state.chats.items())):
        title = chat["title"]
        if len(title) > 30:
            title = title[:27] + "..."

        prefix = "●  " if chat_id == st.session_state.active_chat_id else "    "

        st.button(
            prefix + title,
            key=f"history_{chat_id}",
            use_container_width=True,
            on_click=open_chat,
            args=(chat_id,),
        )

    with st.expander("Prompt library"):
        prompt_library = [
            "What is data science?",
            "Explain LangChain simply",
            "Help me write Python code",
            "What can you help me with?",
        ]

        for prompt_text in prompt_library:
            st.button(
                prompt_text,
                key=f"library_{prompt_text}",
                use_container_width=True,
                on_click=queue_prompt,
                args=(prompt_text,),
            )


active_chat = st.session_state.chats[st.session_state.active_chat_id]

if not active_chat["messages"]:
    st.markdown(
        """
        <div class="welcome">
            <div class="welcome-icon">🤖</div>
            <h1>Welcome to Ava ChatBot</h1>
            <p>Choose a prompt below or ask Ava your own question.</p>
        </div>
        """,
        unsafe_allow_html=True,
    )

    suggestions = [
        "What is data science?",
        "Explain LangChain simply",
        "Help me write Python code",
        "What can you help me with?",
    ]

    first_row = st.columns(2)
    second_row = st.columns(2)

    for column, prompt_text in zip(first_row + second_row, suggestions):
        with column:
            st.button(
                prompt_text,
                key=f"suggestion_{prompt_text}",
                use_container_width=True,
                on_click=queue_prompt,
                args=(prompt_text,),
            )
else:
    for message in active_chat["messages"]:
        avatar = "👤" if message["role"] == "user" else "🤖"
        with st.chat_message(message["role"], avatar=avatar):
            st.markdown(message["content"])


with st.spinner("Loading Ava..."):
    chatbot = load_chatbot()


typed_prompt = st.chat_input(
    "Type your message to Ava...",
    max_chars=2000,
)

suggested_prompt = st.session_state.pop("pending_prompt", None)
user_prompt = typed_prompt or suggested_prompt

if user_prompt:
    active_chat["messages"].append(
        {"role": "user", "content": user_prompt}
    )

    if active_chat["title"] == "New chat":
        active_chat["title"] = user_prompt.replace("\n", " ")[:30]

    with st.chat_message("user", avatar="👤"):
        st.markdown(user_prompt)

    model_messages = [
        {
            "role": "system",
            "content": (
                "You are Ava, a friendly and concise assistant. "
                "Answer the user's latest message directly. "
                "Reply naturally to greetings. Do not summarize the chat "
                "unless asked. Do not invent details about the user. "
                "If you do not know, say so."
            ),
        },
        *active_chat["messages"][-8:],
    ]

    formatted_prompt = chatbot.tokenizer.apply_chat_template(
        model_messages,
        tokenize=False,
        add_generation_prompt=True,
    )

    with st.chat_message("assistant", avatar="🤖"):
        with st.spinner("Ava is thinking..."):
            result = chatbot(
                formatted_prompt,
                max_new_tokens=100,
                do_sample=False,
                return_full_text=False,
                eos_token_id=chatbot.tokenizer.eos_token_id,
            )

            reply = result[0]["generated_text"].strip()

            for marker in ("<|user|>", "<|assistant|>", "</s>"):
                reply = reply.split(marker, 1)[0].strip()

            st.markdown(reply)

    active_chat["messages"].append(
        {"role": "assistant", "content": reply}
    )