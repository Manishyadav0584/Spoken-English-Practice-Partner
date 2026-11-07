# Report draft — save/export as PDF named 28_Manish_Yadav_P28.pdf
(Section order required by the doc. Replace every <FILL> with your real output, screenshots and numbers.)

## 1. Cover page
Generative AI CA Mini-Project · P28 Spoken English Practice Partner · Name: Manish Yadav · Roll No: 28 · MIT School of Computing, Pune
Live app: <FILL> · GitHub: <FILL> · Demo video: <FILL>

## 2. Problem and objective
Many learners understand English but lack a safe partner to practise speaking. Goal: an app where the learner speaks, an LLM answers in role (interviewer, shopkeeper, friend) and a coach layer gives gentle grammar corrections, with spoken replies.

## 3. Tools, model and dataset
Python, PyTorch/Transformers, Gradio, Hugging Face Spaces. Whisper-base (ASR), BERT-CoLA (grammar), Llama-3.1-8B-Instant via Groq (dialogue), gTTS (speech), sd-turbo (scene images). No training dataset; CoLA is the pre-fine-tuned model. Evaluation sets: 10 erroneous + 6 correct sentences; 10 sentences x 4 accents.

## 4. Approach with block diagram
Insert the flowchart from README (export from mermaid.live as an image).
- Generative vs discriminative: Whisper, LLM and the diffusion model are generative; BERT-CoLA is discriminative.
- Lifecycle: problem → model selection → prompting/evaluation → deployment → monitoring.
- U1: role prompts, zero-shot vs few-shot, probability view = CoLA P(acceptable).
- U2: transformer LLM; prompt-based learning compared with fine-tuned BERT-CoLA; attention heatmap.
- U3: GPT-style LLM + BERT + Whisper audio. U4: PyTorch stack, TTS, diffusion scenes. U5: see section 9.

## 5. Key code + GitHub link
Show core.py snippets: build_messages(), cola_score(), transcribe(), guard(). Link: <FILL>

## 6. Captioned output screenshots
Fig 1 Interviewer role · Fig 2 Shopkeeper · Fig 3 Friend · Fig 4 grammar table · Fig 5 attention heatmap · Fig 6 deployed URL in browser. <FILL>

## 7. Results table
Required deliverables: (a) one 3-turn conversation with corrections (paste the chat screenshot + text); (b) accent accuracy table from results/accent_wer.csv; (c) prompting table from results/prompting.csv.
| Accent group | Clips | WER % |
|---|---|---|
| <FILL> | | |

## 8. Limitations and failure cases
Document at least one real failure, e.g. a mis-heard word from a fast/accented sentence, or a correct sentence the LLM "corrected". <FILL with your actual example>

## 9. Ethical note
Bias: speech recognition accuracy varies by accent (table above). Privacy: recordings deleted after transcription; PII redaction. IP: model licences listed in MODEL_CARD.md; AI-generated images. Guardrail: unsafe-topic redirect + role-limited prompt.

## 10. Conclusion (3–5 lines)
<FILL after results>
