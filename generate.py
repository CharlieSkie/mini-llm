import torch
import re
from model import MiniGPT

device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Using device: {device}")

checkpoint = torch.load("checkpoints/mini_llm.pt", map_location=device)

stoi = checkpoint["stoi"]
itos = checkpoint["itos"]
vocab_size = checkpoint["vocab_size"]
config = checkpoint["config"]
NOT_IN_TRAIN_MESSAGE = "not in my train"

def encode(s):
    return [stoi[c] for c in s if c in stoi]

def decode(tokens):
    return "".join([itos[i] for i in tokens])

def normalize_question(text: str) -> str:
    text = re.sub(r"^\s*question:\s*", "", text, flags=re.IGNORECASE)
    text = re.sub(r"[^\w\s]", "", text.lower())
    return re.sub(r"\s+", " ", text).strip()

def load_training_answers(filename: str) -> dict[str, str]:
    try:
        with open(filename, "r", encoding="utf-8") as f:
            lines = f.read().replace("\r", "").splitlines()
    except FileNotFoundError:
        return {}

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

def build_model_prompt(prompt: str) -> str:
    lowered = prompt.lower()
    question_starters = ("what ", "why ", "how ", "when ", "where ", "who ", "which ")

    if lowered.startswith(("topic:", "question:", "answer:")):
        return prompt

    if prompt.endswith("?") or lowered.startswith(question_starters):
        return f"Question: {prompt}\nAnswer:"

    return f"Topic: {prompt}\n"

def is_question_prompt(prompt: str) -> bool:
    cleaned = prompt.strip()
    lowered = cleaned.lower()
    question_starters = ("what ", "why ", "how ", "when ", "where ", "who ", "which ")

    if lowered.startswith("question:"):
        return True

    return cleaned.endswith("?") or lowered.startswith(question_starters)

def clean_generated_text(full_output: str, full_prompt: str) -> str:
    if full_output.startswith(full_prompt):
        reply = full_output[len(full_prompt):].strip()
    else:
        reply = full_output.strip()

    marker_match = re.search(
        r"(?:^|\n)\s*(?:Topic|Question|Answer|User|You|Agent):",
        reply,
        flags=re.IGNORECASE,
    )
    if marker_match:
        reply = reply[:marker_match.start()].strip()

    return reply if reply else "..."

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

prompt = input("Enter prompt: ").strip()
training_answers = load_training_answers("data/train.txt")

if prompt:
    training_answer = training_answers.get(normalize_question(prompt))
    if training_answer:
        print("\n=== GENERATED TEXT ===\n")
        print(training_answer)
        raise SystemExit

    if is_question_prompt(prompt):
        print("\n=== GENERATED TEXT ===\n")
        print(NOT_IN_TRAIN_MESSAGE)
        raise SystemExit

    full_prompt = build_model_prompt(prompt)
    encoded = encode(full_prompt)
    if not encoded:
        print("Prompt has no known characters from training data.")
        raise SystemExit
    context = torch.tensor([encoded], dtype=torch.long, device=device)
else:
    full_prompt = ""
    context = torch.zeros((1, 1), dtype=torch.long, device=device)

generated = model.generate(
    context,
    max_new_tokens=300,
    temperature=0.4
)[0].tolist()

print("\n=== GENERATED TEXT ===\n")
print(clean_generated_text(decode(generated), full_prompt))
