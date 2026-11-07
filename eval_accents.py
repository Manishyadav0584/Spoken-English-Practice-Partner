"""U5: measure Whisper word-error-rate (WER) across accents. Writes results/accent_wer.csv + prints a markdown table.
1) Synthetic proxy: gTTS speaks the same sentences with US / UK / Indian / Australian English voices.
2) Real speech (better evidence): put your own/friends' recordings in recordings/<Accent>/NN.wav (NN matches sentence number);
   they are only read locally and should NOT be committed to GitHub."""
import os, glob, csv, string, tempfile
import jiwer, core

SENTENCES = [
    "I would like to apply for the software engineer position.",
    "Could you please tell me the price of these tomatoes?",
    "Yesterday I went to the library with my friends.",
    "What time does the next train to Pune leave?",
    "I am interested in machine learning and data science.",
    "My favourite subject at college is artificial intelligence.",
    "Can you recommend a good restaurant near the station?",
    "We are planning to watch a movie this weekend.",
    "She has been working in this company for three years.",
    "Thank you very much for your help today.",
]
SYNTH = {"US": "com", "UK": "co.uk", "India": "co.in", "Australia": "com.au"}

def norm(s):
    return s.lower().translate(str.maketrans("", "", string.punctuation)).strip()

def wer(ref, hyp):
    return jiwer.wer(norm(ref), norm(hyp))

rows = []
for accent, tld in SYNTH.items():
    errs = []
    for ref in SENTENCES:
        path = core.speak(ref, tld)
        try:
            errs.append(wer(ref, core.transcribe(path)))
        finally:
            os.remove(path)
    rows.append(["synthetic-" + accent, len(errs), round(100 * sum(errs) / len(errs), 1)])

for folder in sorted(glob.glob("recordings/*")):
    errs = []
    for i, ref in enumerate(SENTENCES, 1):
        for f in glob.glob(f"{folder}/{i:02d}.*"):
            errs.append(wer(ref, core.transcribe(f)))
    if errs:
        rows.append(["real-" + os.path.basename(folder), len(errs), round(100 * sum(errs) / len(errs), 1)])

os.makedirs("results", exist_ok=True)
with open("results/accent_wer.csv", "w", newline="") as f:
    csv.writer(f).writerows([["group", "n_clips", "WER_%"]] + rows)
print("| Accent group | Clips | WER % |\n|---|---|---|")
for r in rows:
    print(f"| {r[0]} | {r[1]} | {r[2]} |")
