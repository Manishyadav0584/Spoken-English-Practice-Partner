"""Core logic for the Spoken English Practice Partner (P28).
Used by app.py (UI) and the eval scripts. Models load lazily so the Space starts fast."""
import os, re, tempfile
import requests, torch
from transformers import pipeline, AutoTokenizer, AutoModelForSequenceClassification

ASR_MODEL = os.getenv("ASR_MODEL", "openai/whisper-base")                 # audio -> text (encoder-decoder, generative)
COLA_MODEL = os.getenv("COLA_MODEL", "textattack/bert-base-uncased-CoLA")  # BERT-family grammar classifier (discriminative)
LOCAL_LLM = os.getenv("LOCAL_LLM", "Qwen/Qwen2.5-0.5B-Instruct")           # GPT-style fallback if no API key
GROQ_MODEL = os.getenv("GROQ_MODEL", "llama-3.1-8b-instant")               # GPT-style generator via free API

_cache = {}
def _lazy(key, fn):
    if key not in _cache:
        _cache[key] = fn()
    return _cache[key]

# ---------------------------------------------------------------- U1/U2: role prompts + dialogue
ROLES = {
    "Interviewer": {
        "prompt": "You are a polite job interviewer at a software company. Ask one interview question at a time and react briefly to the candidate's answer.",
        "tld": "co.uk",
        "scene": "a bright modern office interview room, two chairs and a desk, friendly atmosphere, digital illustration",
    },
    "Shopkeeper": {
        "prompt": "You are a friendly shopkeeper in a small neighbourhood grocery store. Help the customer, mention prices in rupees, and ask follow-up questions.",
        "tld": "co.in",
        "scene": "a colourful Indian neighbourhood grocery shop with fruits and vegetables on display, warm light, digital illustration",
    },
    "Friend": {
        "prompt": "You are a cheerful college friend having a casual chat over coffee. Keep the conversation natural and ask about the learner's day and interests.",
        "tld": "com",
        "scene": "two friends chatting at a cosy campus cafe with coffee cups, warm afternoon light, digital illustration",
    },
}

RULES = (
    "You are a friendly English-speaking coach and conversation partner. "
    "Stay in the selected role and respond naturally. "
    "Your response MUST always contain two sections in this exact format:\n\n"
    "REPLY: [Your natural in-character response, maximum 3 short sentences.]\n"
    "CORRECTION: [The learner's complete corrected sentence, followed by a short explanation of the grammar mistake.]\n\n"
    "If the learner's English is already correct, write exactly: CORRECTION: none\n"
    "Never omit the CORRECTION section. "
    "Correct grammar, verb tense, articles, prepositions, spelling and word order. "
    "Do not invent mistakes. Be encouraging and never harsh. "
    "Keep the conversation appropriate and focused on English practice."
)

FEW_SHOT = [
    {
        "role": "user",
        "content": "Yesterday I go to market and buyed two kilo apples."
    },
    {
        "role": "assistant",
        "content": (
            "REPLY: That sounds nice! What kind of apples did you buy?\n"
            "CORRECTION: Yesterday I went to the market and bought two kilos of apples. "
            "Use past-tense verbs 'went' and 'bought' when talking about yesterday."
        )
    },
    {
        "role": "user",
        "content": "I like green apples because they are crunchy."
    },
    {
        "role": "assistant",
        "content": (
            "REPLY: Green apples are delicious! Do you eat them often?\n"
            "CORRECTION: none"
        )
    }
]

def build_messages(role, history, user_text, mode="zero-shot"):
    system = ROLES[role]["prompt"] + " " + RULES
    msgs = [{"role": "system", "content": system}]
    if mode == "few-shot":
        msgs += FEW_SHOT
    msgs += history + [{"role": "user", "content": user_text}]
    return msgs

def llm_chat(messages, temperature=0.4, max_tokens=200):
    key = os.getenv("GROQ_API_KEY")  # set as a Space secret, never in code
    if key:
        try:
            r = requests.post(
                "https://api.groq.com/openai/v1/chat/completions",
                headers={"Authorization": f"Bearer {key}"},
                json={"model": GROQ_MODEL, "messages": messages, "temperature": temperature, "max_tokens": max_tokens},
                timeout=30)
            r.raise_for_status()
            return r.json()["choices"][0]["message"]["content"].strip()
        except Exception as e:
            print("API call failed, using local model:", type(e).__name__)
    gen = _lazy("llm", lambda: pipeline("text-generation", model=LOCAL_LLM, device=-1))
    out = gen(messages, max_new_tokens=max_tokens, do_sample=temperature > 0, temperature=max(temperature, 0.01))
    return out[0]["generated_text"][-1]["content"].strip()

def parse_reply(raw):
    parts = re.split(r"CORRECTION:", raw, maxsplit=1, flags=re.I)
    reply = parts[0].strip()
    correction = parts[1].strip() if len(parts) > 1 else "none"
    return reply or raw, correction

# ---------------------------------------------------------------- U3: Whisper + BERT (CoLA)
def transcribe(path):
    asr = _lazy("asr", lambda: pipeline("automatic-speech-recognition", model=ASR_MODEL, device=-1))
    out = asr(path, chunk_length_s=30, generate_kwargs={"language": "english", "task": "transcribe"})
    return out["text"].strip()

def _cola():
    def load():
        tok = AutoTokenizer.from_pretrained(COLA_MODEL)
        mdl = AutoModelForSequenceClassification.from_pretrained(COLA_MODEL, attn_implementation="eager").eval()
        return tok, mdl
    return _lazy("cola", load)

def split_sentences(text):
    return [s.strip() for s in re.split(r"(?<=[.!?])\s+", text.strip()) if s.strip()]

@torch.no_grad()
def cola_score(sentence):
    """P(grammatically acceptable). CoLA label 1 = acceptable."""
    tok, mdl = _cola()
    enc = tok(sentence, return_tensors="pt", truncation=True, max_length=64)
    return float(torch.softmax(mdl(**enc).logits, -1)[0][1])

@torch.no_grad()
def cola_attention(sentence):
    """Last-layer attention averaged over heads -> (tokens, matrix) for the U2 attention explanation."""
    tok, mdl = _cola()
    enc = tok(sentence, return_tensors="pt", truncation=True, max_length=64)
    out = mdl(**enc, output_attentions=True)
    att = out.attentions[-1][0].mean(0).numpy()
    return tok.convert_ids_to_tokens(enc["input_ids"][0]), att

# ---------------------------------------------------------------- U4: TTS
def speak(text, tld="com"):
    from gtts import gTTS
    f = tempfile.NamedTemporaryFile(suffix=".mp3", delete=False)
    f.close()
    gTTS(text=text, lang="en", tld=tld).save(f.name)
    return f.name

# ---------------------------------------------------------------- U5: guardrails
BLOCKED = re.compile(r"\b(kill yourself|suicide|bomb|terroris[mt]|porn|nude|credit card number|password)\b", re.I)
PII = [(re.compile(r"[\w.+-]+@[\w-]+\.[\w.]+"), "[EMAIL]"), (re.compile(r"\b\d{10,}\b"), "[NUMBER]")]

def guard(text):
    """Returns (safe_text, refusal_message_or_None). Redacts email/phone-like numbers, blocks unsafe topics."""
    if BLOCKED.search(text):
        return text, "Let's keep our practice friendly and on topic. Could you try saying something about your day, work or hobbies?"
    for rx, tag in PII:
        text = rx.sub(tag, text)
    return text, None
