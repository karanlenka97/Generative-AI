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
            "You are a friendly, concise assistant. Reply naturally to "
            "greetings. Answer the user's latest message directly. Do not "
            "summarize the conversation unless asked. Do not invent details "
            "about the user or previous conversations. If you do not know "
            "something, say so."
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


print(f"\n{CYAN}╭────────────────────────────────────╮")
print("│       TinyLlama Terminal Chat      │")
print(f"╰────────────────────────────────────╯{RESET}")
print(f"{YELLOW}Type 0 and press Enter to close.{RESET}\n")

while True:
    prompt = input(f"{GREEN}You › {RESET}").strip()

    if prompt == "0":
        print(f"{YELLOW}Chat closed.{RESET}")
        break

    if not prompt:
        continue

    conversation.append(HumanMessage(content=prompt))

    recent_history = [conversation[0]] + conversation[1:][-8:]
    model_messages = to_transformers_messages(recent_history)

    formatted_prompt = chatbot.tokenizer.apply_chat_template(
        model_messages,
        tokenize=False,
        add_generation_prompt=True,
    )

    result = chatbot(
        formatted_prompt,
        max_new_tokens=80,
        do_sample=False,
        return_full_text=False,
        eos_token_id=chatbot.tokenizer.eos_token_id,
    )

    reply = result[0]["generated_text"].strip()

    for marker in ("<|user|>", "<|assistant|>", "</s>"):
        reply = reply.split(marker, 1)[0].strip()

    conversation.append(AIMessage(content=reply))
    print(f"{CYAN}TinyLlama › {RESET}{reply}\n")

print("\nFull conversation:")
for message in conversation[1:]:
    if isinstance(message, HumanMessage):
        print(f"You: {message.content}")
    elif isinstance(message, AIMessage):
        print(f"TinyLlama: {message.content}")