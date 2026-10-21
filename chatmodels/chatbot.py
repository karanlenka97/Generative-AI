from transformers import pipeline
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage

CYAN = "\033[96m"
GREEN = "\033[92m"
YELLOW = "\033[93m"
RESET = "\033[0m"

chatbot = pipeline(
    "text-generation",
    model="TinyLlama/TinyLlama-1.1B-Chat-v1.0",
)

conversation = [
    SystemMessage(
        content=(
            "Answer briefly and directly. Use the conversation history "
            "to remember details the user has shared."
        )
    )
]

def to_transformers_messages(messages):
    converted = []

    for message in messages:
        if isinstance(message, SystemMessage):
            role = "system"
        elif isinstance(message, HumanMessage):
            role = "user"
        elif isinstance(message, AIMessage):
            role = "assistant"
        else:
            continue

        converted.append(
            {"role": role, "content": str(message.content)}
        )

    return converted

while True:
    prompt = input(f"{GREEN}You › {RESET}").strip()

    if prompt == "0":
        print(f"{YELLOW}Chat closed.{RESET}")
        break

    if not prompt:
        continue

    conversation.append(HumanMessage(content=prompt))

    recent_history = [conversation[0]] + conversation[1:][-12:]
    model_messages = to_transformers_messages(recent_history)

    result = chatbot(
        model_messages,
        max_new_tokens=80,
        do_sample=True,
        temperature=0.3,
        top_p=0.9,
        repetition_penalty=1.1,
    )

    reply = result[0]["generated_text"][-1]["content"]
    conversation.append(AIMessage(content=reply))

    print(f"{CYAN}TinyLlama › {RESET}{reply}\n")