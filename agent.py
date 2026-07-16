import os
import re
import sys
import time
import torch
from model import MiniGPT

# =========================
# Device
# =========================
device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Using device: {device}")

# =========================
# Load checkpoint
# =========================
checkpoint = torch.load("checkpoints/mini_llm.pt", map_location=device)

stoi = checkpoint["stoi"]
itos = checkpoint["itos"]
vocab_size = checkpoint["vocab_size"]
config = checkpoint["config"]

# =========================
# Encode / Decode
# =========================
def encode(s: str):
    return [stoi[c] for c in s if c in stoi]

def decode(tokens):
    return "".join([itos[i] for i in tokens])

# =========================
# Load model
# =========================
model = MiniGPT(
    vocab_size=vocab_size,
    n_embd=config["n_embd"],
    block_size=config["block_size"],
    n_head=config["n_head"],
    n_layer=config["n_layer"],
    dropout=config["dropout"],
).to(device)

model.load_state_dict(checkpoint["model_state_dict"])
model.eval()

# =========================
# Settings
# =========================
MAX_NEW_TOKENS = 700  # Allow longer multi-line/code responses
TEMPERATURE = 0.4  # Lower temperature gives more stable CPU-trained output
MAX_HISTORY_CHARS = 1000
TYPE_DELAY = 0.010  # mas mababa = mas mabilis mag-type

stop_marker_pattern = re.compile(
    r"(?:^|\n)\s*(?:Topic|Question|Answer|User|You|Agent):",
    re.IGNORECASE,
)

# =========================
# Helpers
# =========================
def clean_text(text: str) -> str:
    text = text.replace("\r", "").strip()
    return text if text else "..."

def normalize_question(text: str) -> str:
    text = re.sub(r"^\s*question:\s*", "", text, flags=re.IGNORECASE)
    text = re.sub(r"[^\w\s]", "", text.lower())
    return re.sub(r"\s+", " ", text).strip()

def load_training_answers(filename: str) -> dict[str, str]:
    if not os.path.exists(filename):
        return {}

    with open(filename, "r", encoding="utf-8") as f:
        lines = f.read().replace("\r", "").splitlines()

    answers = {}
    question = ""
    answer_lines = []

    def save_current_answer():
        normalized = normalize_question(question)
        answer = "\n".join(answer_lines).strip()
        if normalized and answer:
            answers[normalized] = answer

    for line in lines:
        lowered = line.lower()

        if lowered.startswith("question:"):
            save_current_answer()
            question = line.split(":", 1)[1].strip()
            answer_lines = []
        elif lowered.startswith("answer:"):
            answer_lines = [line.split(":", 1)[1].strip()]
        elif lowered.startswith("topic:"):
            save_current_answer()
            question = ""
            answer_lines = []
        elif question and answer_lines:
            if line.strip():
                answer_lines.append(line)
            else:
                save_current_answer()
                question = ""
                answer_lines = []

    save_current_answer()
    return answers

training_answers = load_training_answers("data/train.txt")
NOT_IN_TRAIN_MESSAGE = "not in my train"

def type_out(text: str, delay: float = TYPE_DELAY):
    """
    Simulated streaming / typing effect.
    """
    for ch in text:
        print(ch, end="", flush=True)
        time.sleep(delay)
    print()

def save_chat_log(chat_history: str, filename: str = "chat_log.txt"):
    with open(filename, "w", encoding="utf-8") as f:
        f.write(chat_history)

def build_model_prompt(prompt: str) -> str:
    """
    Match the training-data style so the model gets a familiar prefix.
    """
    lowered = prompt.lower()
    known_prefixes = (
        "topic:",
        "question:",
        "answer:",
        "problem:",
        "explanation:",
        "solution:",
        "analysis:",
        "concept:",
    )
    question_starters = ("what ", "why ", "how ", "when ", "where ", "who ", "which ")

    if lowered.startswith(known_prefixes):
        return prompt

    if prompt.endswith("?") or lowered.startswith(question_starters):
        return f"Question: {prompt}\nAnswer:"

    return f"Topic: {prompt}\n"

def is_question_prompt(prompt: str) -> bool:
    """
    Detect normal user questions, including prompts already written in train.txt style.
    """
    cleaned = prompt.strip()
    lowered = cleaned.lower()
    question_starters = ("what ", "why ", "how ", "when ", "where ", "who ", "which ")

    if lowered.startswith("question:"):
        return True

    return cleaned.endswith("?") or lowered.startswith(question_starters)

def generate_response(prompt: str, history: str = "") -> str:
    """
    Generate text based on the prompt using a format similar to data/train.txt.
    """
    training_answer = training_answers.get(normalize_question(prompt))
    if training_answer:
        return clean_text(training_answer)

    if is_question_prompt(prompt):
        return NOT_IN_TRAIN_MESSAGE

    full_prompt = build_model_prompt(prompt)

    encoded = encode(full_prompt)
    if not encoded:
        return "I cannot understand this prompt because the characters were not in the training data."

    context = torch.tensor([encoded], dtype=torch.long, device=device)

    with torch.no_grad():
        generated = model.generate(
            context,
            max_new_tokens=MAX_NEW_TOKENS,
            temperature=TEMPERATURE
        )[0].tolist()

    full_output = decode(generated)
    
    # Remove the formatted model prompt from the output
    if full_output.startswith(full_prompt):
        reply = full_output[len(full_prompt):].strip()
    else:
        reply = full_output.strip()
    
    # Stop when the model starts another training/chat block.
    marker_match = stop_marker_pattern.search(reply)
    if marker_match:
        reply = reply[:marker_match.start()].strip()

    # Limit very long output without destroying normal code formatting.
    if len(reply) > 1200:
        reply = reply[:1200].rstrip() + "..."

    return clean_text(reply)

# =========================
# Intro
# =========================
print("\nMini LLM Agent")
print("Commands:")
print("  /exit   - quit")
print("  /clear  - clear memory")
print("  /save   - save chat")
print("  /fast   - faster typing")
print("  /slow   - slower typing")
print("  /temp X - set temperature, example: /temp 0.8\n")

chat_history = ""

# =========================
# Chat loop
# =========================
while True:
    user_input = input("You: ").strip()

    if not user_input:
        continue

    if user_input.lower() == "/exit":
        print("Agent: Goodbye.")
        break

    if user_input.lower() == "/clear":
        chat_history = ""
        print("Agent: Memory cleared.")
        continue

    if user_input.lower() == "/save":
        save_chat_log(chat_history)
        print("Agent: Chat saved to chat_log.txt")
        continue

    if user_input.lower() == "/fast":
        TYPE_DELAY = 0.005
        print("Agent: Typing speed set to fast.")
        continue

    if user_input.lower() == "/slow":
        TYPE_DELAY = 0.03
        print("Agent: Typing speed set to slow.")
        continue

    if user_input.lower().startswith("/temp "):
        try:
            value = float(user_input.split(" ", 1)[1])
            if value <= 0:
                print("Agent: Temperature must be greater than 0.")
                continue
            TEMPERATURE = value
            print(f"Agent: Temperature set to {TEMPERATURE}")
        except ValueError:
            print("Agent: Invalid temperature value.")
        continue

    response = generate_response(user_input, history=chat_history)

    print("Agent: ", end="", flush=True)
    type_out(response, TYPE_DELAY)
    print()

    # Store simpler history - just for logging, not used in generation
    chat_history += f"User: {user_input}\nAgent: {response}\n"

    if len(chat_history) > MAX_HISTORY_CHARS:
        chat_history = chat_history[-MAX_HISTORY_CHARS:]
