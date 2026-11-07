# Model card: Spoken English Practice Partner

**Intended use:** low-stakes spoken English practice for learners; not for exams, hiring decisions or certifying proficiency.

**Components (verify each licence on its Hugging Face page before submitting):**
| Component | Model | Role |
|---|---|---|
| Speech recognition | openai/whisper-base | speech to text |
| Grammar classifier | textattack/bert-base-uncased-CoLA | P(acceptable) per sentence |
| Dialogue LLM | Llama-3.1-8B-Instant via Groq API (or Qwen2.5-0.5B-Instruct locally) | role-play + corrections |
| TTS | gTTS (Google Translate TTS) | spoken replies |
| Scene images | stabilityai/sd-turbo | one image per role |

**Data:** no training in this project; models are used as released. Evaluation sentences are hand-written (`eval_prompting.py`, `eval_accents.py`).

**Results:** see `results/` (accent WER table, prompting comparison). <PASTE YOUR NUMBERS>

**Limitations:** CoLA judges grammatical acceptability, not meaning or pronunciation; Whisper-base makes more errors on accented and noisy speech; the LLM can "over-correct" correct sentences or give a wrong tip; gTTS needs internet and has fixed voices; synthetic accent tests are only a proxy for real speakers.

**Ethics and privacy:** audio is deleted right after transcription and never stored; email/phone-like numbers are redacted before the LLM call; unsafe topics get a polite redirect; no personal data is logged. Accent-based accuracy gaps are measured and reported, not hidden.

**Licences / IP:** Whisper (MIT), BERT-CoLA and gTTS (check pages), Llama 3.1 community licence, sd-turbo (check its non-commercial terms). Generated images and replies are AI-generated; the project is for education only.
