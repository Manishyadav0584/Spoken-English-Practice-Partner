"""U1/U2/U5: (a) zero-shot vs few-shot prompting for sentence correction, (b) prompt-based LLM vs fine-tuned BERT-CoLA
at spotting errors, (c) false-correction rate on already-correct sentences (a simple hallucination test)."""
import re, string, difflib, csv, os
import core

WRONG = [
    ("I goes to market yesterday.", "I went to the market yesterday."),
    ("She don't like coffee.", "She doesn't like coffee."),
    ("He have two brother.", "He has two brothers."),
    ("Me and my friend is going to cinema.", "My friend and I are going to the cinema."),
    ("I am agree with you.", "I agree with you."),
    ("What time you will come?", "What time will you come?"),
    ("She is more taller than me.", "She is taller than me."),
    ("I have seen him yesterday.", "I saw him yesterday."),
    ("They was playing cricket in the park.", "They were playing cricket in the park."),
    ("Can you give me an informations about the course?", "Can you give me some information about the course?"),
]
RIGHT = [
    "I went to the market yesterday.", "She doesn't like coffee.", "My friend and I are going to the cinema.",
    "What time will you come?", "They were playing cricket in the park.", "I would like to apply for the position.",
]
SHOTS = [
    {"role": "user", "content": "He go to school every days."}, {"role": "assistant", "content": "He goes to school every day."},
    {"role": "user", "content": "I want to learn English."}, {"role": "assistant", "content": "I want to learn English."},
    {"role": "user", "content": "Where you are living now?"}, {"role": "assistant", "content": "Where are you living now?"},
]
SYS = "Correct the grammar of the user's sentence. If it is already correct, repeat it unchanged. Reply with ONLY the sentence."

def norm(s): return re.sub(r"\s+", " ", s.lower().translate(str.maketrans("", "", string.punctuation))).strip()
def fix(sentence, few):
    msgs = [{"role": "system", "content": SYS}] + (SHOTS if few else []) + [{"role": "user", "content": sentence}]
    return core.llm_chat(msgs, temperature=0.0, max_tokens=60).strip().split("\n")[0]

rows = []
for name, few in [("zero-shot", False), ("few-shot", True)]:
    exact = sim = 0
    for bad, gold in WRONG:
        out = fix(bad, few); exact += norm(out) == norm(gold); sim += difflib.SequenceMatcher(None, norm(out), norm(gold)).ratio()
    false_corr = sum(norm(fix(s, few)) != norm(s) for s in RIGHT)
    rows.append([name, f"{exact}/{len(WRONG)}", round(sim / len(WRONG), 2), f"{false_corr}/{len(RIGHT)}"])

cola_ok = sum(core.cola_score(b) < 0.5 for b, _ in WRONG) + sum(core.cola_score(s) >= 0.5 for s in RIGHT)
os.makedirs("results", exist_ok=True)
with open("results/prompting.csv", "w", newline="") as f:
    csv.writer(f).writerows([["technique", "exact_fix", "avg_similarity", "false_corrections"]] + rows)
print("| Technique | Exact fixes | Avg similarity | False corrections (on correct input) |\n|---|---|---|---|")
for r in rows: print("| " + " | ".join(map(str, r)) + " |")
print(f"\nBERT-CoLA (fine-tuned classifier) judged {cola_ok}/{len(WRONG)+len(RIGHT)} sentences correctly as acceptable/unacceptable.")
