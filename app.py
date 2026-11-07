
import os

import gradio as gr
import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import core


ASSETS = os.path.join(os.path.dirname(__file__), "assets")


def scene_for(role):
    """Load the scene image assigned to the selected role."""
    image_path = os.path.join(
        ASSETS,
        f"scene_{role.lower()}.png"
    )

    return image_path if os.path.exists(image_path) else None


def attention_fig(sentence):
    """Create a BERT attention visualization."""
    tokens, attention = core.cola_attention(sentence)

    fig, ax = plt.subplots(figsize=(5, 4))

    ax.imshow(attention, cmap="viridis")

    ax.set_xticks(range(len(tokens)))
    ax.set_xticklabels(
        tokens,
        rotation=90,
        fontsize=7
    )

    ax.set_yticks(range(len(tokens)))
    ax.set_yticklabels(tokens, fontsize=7)

    ax.set_title(
        "BERT (CoLA) last-layer attention",
        fontsize=9
    )

    fig.tight_layout()

    return fig


def respond(audio, typed, role, mode, state):

    state = state or {
        "llm": [],
        "ui": []
    }

    transcript = ""

    # --------------------------------
    # 1. Speech transcription (Whisper)
    # --------------------------------

    if audio:
        try:
            transcript = core.transcribe(audio)

        finally:
            # Delete uploaded/recorded audio after transcription
            try:
                os.remove(audio)
            except OSError:
                pass

    text = transcript or (typed or "").strip()

    if not text:
        return (
            state["ui"],
            "",
            [],
            None,
            scene_for(role),
            None,
            state
        )

    # --------------------------------
    # 2. Safety / input guard
    # --------------------------------

    text, refusal = core.guard(text)

    # --------------------------------
    # 3. BERT-CoLA grammar scoring
    # --------------------------------

    sentences = core.split_sentences(text) or [text]

    scores = [
        core.cola_score(sentence)
        for sentence in sentences
    ]

    rows = [
        [
            sentence,
            round(score, 3),
            "OK" if score >= 0.5 else "check"
        ]
        for sentence, score in zip(sentences, scores)
    ]

    worst_sentence = sentences[scores.index(min(scores))]

    # --------------------------------
    # 4. LLM conversation and correction
    # --------------------------------

    if refusal:

        reply = refusal
        correction = "none"

    else:

        messages = core.build_messages(
            role,
            state["llm"],
            text,
            mode
        )

        raw = core.llm_chat(messages)

        # Retry if the model forgets the correction section
        if "CORRECTION:" not in raw.upper():

            retry_messages = messages + [
                {
                    "role": "assistant",
                    "content": raw
                },
                {
                    "role": "user",
                    "content": (
                        "Rewrite your previous answer using the required format. "
                        "You MUST include both REPLY: and CORRECTION: sections. "
                        "Correct the original learner sentence: " + text
                    )
                }
            ]

            raw = core.llm_chat(
                retry_messages,
                temperature=0.1
            )

        reply, correction = core.parse_reply(raw)

        # Preserve the complete correction context
        state["llm"] += [
            {
                "role": "user",
                "content": text
            },
            {
                "role": "assistant",
                "content": (
                    f"REPLY: {reply}\n"
                    f"CORRECTION: {correction}"
                )
            }
        ]

    # --------------------------------
    # 5. Display conversation
    # --------------------------------

    if correction.lower().startswith("none"):
        shown = reply

    else:
        shown = (
            f"{reply}\n\n"
            f"✏️ {correction}"
        )

    state["ui"] += [
        {
            "role": "user",
            "content": text
        },
        {
            "role": "assistant",
            "content": shown
        }
    ]

    # --------------------------------
    # 6. Text-to-speech
    # --------------------------------

    audio_out = core.speak(
        reply,
        core.ROLES[role]["tld"]
    )

    # --------------------------------
    # 7. Scene and attention visualization
    # --------------------------------

    scene = scene_for(role)
    attention = attention_fig(worst_sentence)

    return (
        state["ui"],
        text,
        rows,
        audio_out,
        scene,
        attention,
        state
    )


def reset():
    """Reset the conversation and interface."""

    return (
        [],
        "",
        [],
        None,
        None,
        None,
        {
            "llm": [],
            "ui": []
        }
    )


# ========================================
# GRADIO USER INTERFACE
# ========================================

with gr.Blocks(
    title="Spoken English Practice Partner"
) as demo:

    gr.Markdown(
        """
        # 🗣️ Spoken English Practice Partner

        Speak or type in English, practise real-life conversations,
        receive grammar feedback and listen to spoken replies.

        **Privacy:** Recordings are deleted after transcription
        and are not intentionally stored by this application.
        """
    )

    state = gr.State(
        {
            "llm": [],
            "ui": []
        }
    )

    # Role and prompting technique
    with gr.Row():

        role = gr.Dropdown(
            choices=list(core.ROLES.keys()),
            value="Friend",
            label="Choose conversation role"
        )

        mode = gr.Radio(
            choices=["zero-shot", "few-shot"],
            value="few-shot",
            label="Prompt technique"
        )

    with gr.Row():

        # Left column
        with gr.Column():

            mic = gr.Audio(
                sources=["microphone", "upload"],
                type="filepath",
                label="Speak or upload audio"
            )

            typed = gr.Textbox(
                label="Or type your sentence",
                placeholder="Enter your English sentence here..."
            )

            with gr.Row():

                go = gr.Button(
                    "Send",
                    variant="primary"
                )

                clr = gr.Button(
                    "New conversation"
                )

            transcript = gr.Textbox(
                label="What I heard (Whisper)",
                interactive=False
            )

            grammar = gr.Dataframe(
                headers=[
                    "Sentence",
                    "P(acceptable)",
                    "Verdict"
                ],
                label="Grammar score (BERT-CoLA)"
            )

        # Right column
        with gr.Column():

            chat = gr.Chatbot(
                type="messages",
                label="Conversation",
                height=320
            )

            voice = gr.Audio(
                label="Spoken reply (TTS)",
                autoplay=True
            )

            scene = gr.Image(
                label="Role scene",
                height=220
            )

            att = gr.Plot(
                label="BERT attention visualization"
            )

    # Component outputs
    outputs = [
        chat,
        transcript,
        grammar,
        voice,
        scene,
        att,
        state
    ]

    # Send message
    go.click(
        fn=respond,
        inputs=[
            mic,
            typed,
            role,
            mode,
            state
        ],
        outputs=outputs
    )

    # Reset conversation
    clr.click(
        fn=reset,
        inputs=None,
        outputs=outputs
    )

    # Update role scene when role changes
    role.change(
        fn=scene_for,
        inputs=role,
        outputs=scene
    )


if __name__ == "__main__":
    demo.launch(server_name="127.0.0.1", server_port=7860)

