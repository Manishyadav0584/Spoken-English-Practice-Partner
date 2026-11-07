# Do in this order (deadline Sat 7 Nov 2026)
1. Create free accounts: Hugging Face, Groq (API key). Verify the 4 model names still exist on huggingface.co.
2. Colab (GPU): upload folder, `pip install -r requirements-dev.txt`, run `generate_scenes.py`, download assets/*.png into the repo.
3. Create HF Space (Gradio), upload all files, add secret GROQ_API_KEY, wait for build, fix any error from the build log.
4. Test the live URL: 3-turn conversation in each role; save screenshots.
5. Run `python eval_prompting.py` and `python eval_accents.py` (Colab or locally); record 5 real clips per accent if you can (do not commit them).
6. Push the same files to a public GitHub repo (no keys, no recordings). Put the Space URL and video link in README.
7. Fill REPORT_DRAFT.md, export PDF as 28_Manish_Yadav_P28.pdf.
8. Record a 2–3 min screen demo (all 5 layers + one failure case).
9. Upload PDF + links + video to the shared-drive folder 28_Manish_Yadav.
