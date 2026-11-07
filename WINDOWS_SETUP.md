# Windows 11 setup (P28)

## 1. Prerequisites
- Python 3.11 (64-bit)
- Git for Windows
- FFmpeg installed and available on PATH (check with `ffmpeg -version`)

## 2. Create an environment
Open PowerShell in this project folder:

```powershell
py -3.11 -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
pip install -r requirements.txt
```

If PowerShell blocks activation, run this once in that terminal:

```powershell
Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass
.\.venv\Scripts\Activate.ps1
```

## 3. Configure the LLM
Recommended: create a Groq API key and set it only in your local PowerShell session:

```powershell
$env:GROQ_API_KEY="paste-your-key-here"
```

Do not put the key in Python files, README, screenshots, or GitHub. Without a key, the app downloads and runs the Qwen fallback model on CPU, which may be slow.

## 4. Run

```powershell
python app.py
```

Open the local URL printed in the terminal (normally http://127.0.0.1:7860).

## 5. Before deployment
- Test typed input first, then microphone input.
- Confirm all three roles work.
- Run `python eval_prompting.py` and `python eval_accents.py` after model downloads finish.
- Never commit API keys or real recordings.
