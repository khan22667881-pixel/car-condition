# car-condition
# 🚗 Car Visible Condition Checker

A Streamlit app that looks at an uploaded photo of a car, together with the make, model and year you enter, and uses **Google Gemini (Flash)** to describe the car's **visible exterior condition**, observable damage or wear, and sensible next steps.

> **Important:** This app is an informal, AI-generated aid. It is **not** a professional inspection, appraisal or valuation. A single photo cannot show hidden mechanical, electrical, structural, flood or accident-repair problems. Always have a qualified mechanic inspect a vehicle before you buy it.

## Features

- Upload a JPG, PNG or WebP photo (up to 10 MB) and enter make, model and year
- Cautious rating: Excellent / Good / Fair / Poor / Cannot determine, with a confidence level that is never higher than "Medium"
- Area-by-area observations, a list of visible damage or wear, and practical follow-up recommendations
- Clear list of what a photo cannot tell you
- Graceful handling of: missing API key, unreadable, corrupted, tiny or oversized images, empty or blocked model responses, malformed model output, rate limits, invalid keys and network errors
- API key is never stored in source code (Streamlit secrets or an environment variable)

## Project files

```
app.py            # the Streamlit app
requirements.txt  # Python dependencies
README.md         # this file
```

## Requirements

- Python 3.10 or newer
- A Google Gemini API key (free tier available): <https://aistudio.google.com/apikey>
- Dependencies (installed from `requirements.txt`):
  - `streamlit` for the web UI
  - `google-genai` for the Gemini API (the package is `google-genai`, **not** `google-generativeai`)
  - `Pillow` for image validation and resizing

## Run locally

1. **Create and activate a virtual environment** (recommended)

   ```bash
   python -m venv .venv
   # macOS / Linux
   source .venv/bin/activate
   # Windows (PowerShell)
   .venv\Scripts\Activate.ps1
   ```

2. **Install dependencies**

   ```bash
   pip install -r requirements.txt
   ```

3. **Configure your API key** (see the next section)

4. **Start the app**

   ```bash
   streamlit run app.py
   ```

   Streamlit opens the app at <http://localhost:8501>.

## Configure the Gemini API key securely

The app looks for the key in this order, and the first one found wins:

1. Streamlit secrets: `GEMINI_API_KEY` (or `GOOGLE_API_KEY`)
2. Environment variable: `GEMINI_API_KEY` (or `GOOGLE_API_KEY`)

### Option A: environment variable (local)

```bash
# macOS / Linux
export GEMINI_API_KEY="your-key-here"

# Windows PowerShell
$env:GEMINI_API_KEY="your-key-here"

# Windows Command Prompt
set GEMINI_API_KEY=your-key-here
```

### Option B: Streamlit secrets file (local)

Create the file `.streamlit/secrets.toml` in the project folder:

```toml
GEMINI_API_KEY = "your-key-here"
```

### Never commit your key

Create a file named `.gitignore` in the project folder with at least these lines **before** your first commit:

```gitignore
.streamlit/secrets.toml
.env
.venv/
__pycache__/
```

If a key is ever pushed to GitHub by mistake, treat it as compromised: delete it in Google AI Studio, create a new one, and update your secrets.

### Optional: choose a different model

The app defaults to `gemini-2.5-flash`. If Google retires or renames that model, or you want a newer Flash model, set `GEMINI_MODEL` the same way as the key (environment variable or secret), for example:

```toml
GEMINI_MODEL = "gemini-2.5-flash"
```

## Push the project to GitHub

1. Make sure you have [Git](https://git-scm.com/downloads) installed and a [GitHub](https://github.com) account.
2. Confirm `.gitignore` exists (see above) and that `.streamlit/secrets.toml` is listed in it.
3. In the project folder, initialise the repository and make the first commit:

   ```bash
   git init
   git add app.py requirements.txt README.md .gitignore
   git commit -m "Initial commit: car visible condition checker"
   git branch -M main
   ```

4. On GitHub, click **New repository**. Choose a name (for example `car-condition-checker`), choose Public or Private, and **do not** add a README, .gitignore or license (your project already has them). Click **Create repository**.
5. Connect your local project to the new repository and push (replace the URL with your own):

   ```bash
   git remote add origin https://github.com/YOUR-USERNAME/car-condition-checker.git
   git push -u origin main
   ```

6. Refresh the GitHub page and confirm that `app.py`, `requirements.txt` and `README.md` are at the **root** of the repository, and that no secrets file was uploaded.

## Deploy on Streamlit Community Cloud

1. Go to <https://share.streamlit.io> and sign in with your GitHub account. Authorise access to your repositories if prompted (private repositories need that permission).
2. Click **Create app** (or **New app**) and choose **Deploy a public app from GitHub** / **From existing repo**.
3. Fill in:
   - **Repository:** `YOUR-USERNAME/car-condition-checker`
   - **Branch:** `main`
   - **Main file path:** `app.py`
   - **App URL:** pick a custom subdomain if you like
4. Click **Advanced settings**:
   - Choose a **Python version** (3.11 or 3.12 works well)
   - In the **Secrets** box, paste:

     ```toml
     GEMINI_API_KEY = "your-key-here"
     ```

5. Click **Deploy**. The first build installs everything from `requirements.txt` and takes a minute or two.
6. Open the app URL and test it with a car photo.

**Updating the app:** commit and push to `main`; Streamlit Community Cloud redeploys automatically.

**Changing the key later:** open the app on <https://share.streamlit.io>, then **Settings -> Secrets**, edit the value and save. The app restarts with the new secret.

## Troubleshooting

| Problem | Likely cause and fix |
| --- | --- |
| `ImportError: cannot import name 'genai' from 'google'` or `ModuleNotFoundError` | A dependency is missing. Make sure `requirements.txt` is at the repo root and contains `google-genai`, `streamlit` and `Pillow`, then reboot the app. |
| "No Gemini API key was found" | The key is not set. Set the environment variable locally, or add it under **Settings -> Secrets** on Streamlit Community Cloud. |
| "The Gemini API rejected the API key" | The key is wrong, deleted or restricted. Create a new one at Google AI Studio. |
| Rate limit or quota message | The free tier has per-minute and per-day limits. Wait and retry, or enable billing. |
| "The configured Gemini model was not found" | Set `GEMINI_MODEL` to a currently available Gemini Flash model. |
| Image rejected | Use a JPG, PNG or WebP under 10 MB and at least 200 px on the short side. HEIC photos from some phones must be converted to JPG first. |
| "The model's answer could not be read" | Rare formatting failure from the model. Click the button again. |

## Limitations

- Results depend on photo quality, lighting, angle and image compression. Reflections and dirt can hide damage or look like damage.
- One photo shows only part of a car. Wheels, underbody, interior and engine bay are not assessed unless visible.
- The AI can be wrong. Treat the output as a starting point for questions, not as an answer.
- The app does not estimate price or value, and it does not store uploaded photos. Images are sent to the Google Gemini API for analysis, so do not upload photos that show faces, licence plates or other personal information you want to keep private.
- Each analysis uses your API quota. If you make the deployed app public, consider usage limits, since anyone with the link can trigger requests on your key.
