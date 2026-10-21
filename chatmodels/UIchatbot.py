import streamlit as st
from transformers import pipeline

st.set_page_config(page_title="TinyLlama Chat", page_icon="💬")
st.title("TinyLlama Chat")

@st.cache_resource
def load_chatbot():
    return pipeline(
        "text-generation",
        model="TinyLlama/TinyLlama-1.1B-Chat-v1.0",
    )

chatbot = load_chatbot()

if "conversation" not in st.session_state:
    st.session_state.conversation = []

# Show the current conversation
for message in st.session_state.conversation:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

prompt = st.chat_input("Type your message")

if prompt:
    st.session_state.conversation.append(
        {"role": "user", "content": prompt}
    )

    with st.chat_message("user"):
        st.markdown(prompt)

    system_message = {
        "role": "system",
        "content": (
            "Answer briefly and directly. Use the conversation history "
            "to remember details the user has shared."
        ),
    }

    model_messages = (
        [system_message]
        + st.session_state.conversation[-12:]
    )

    with st.chat_message("assistant"):
        with st.spinner("Thinking..."):
            result = chatbot(
                model_messages,
                max_new_tokens=80,
                do_sample=True,
                temperature=0.3,
                top_p=0.9,
                repetition_penalty=1.1,
            )
            reply = result[0]["generated_text"][-1]["content"]
            st.markdown(reply)

    st.session_state.conversation.append(
        {"role": "assistant", "content": reply}
    )