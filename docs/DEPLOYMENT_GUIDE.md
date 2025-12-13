# Free Deployment Guide for Stock Prediction Webapp

This guide explains how to deploy your stock prediction webapp online for **free** so teammates can test it.

## Quick Comparison

| Platform | Free Tier | Cold Start | Best For |
|----------|-----------|------------|----------|
| **Render** | 750 hrs/month | ~30s | Easy setup, reliable |
| **Railway** | $5 credit/month | ~5s | Fast iteration |
| **PythonAnywhere** | 1 app, always on | None | No cold starts |
| **Hugging Face** | Unlimited | ~20s | ML demos |

---

## Option 1: Render.com (Recommended)

### Step 1: Push to GitHub
```bash
# In your project folder
git init
git add .
git commit -m "Initial commit"
git remote add origin https://github.com/YOUR_USERNAME/stock-prediction.git
git push -u origin main
```

### Step 2: Deploy on Render
1. Go to [render.com](https://render.com) and sign up (free)
2. Click **New** → **Web Service**
3. Connect your GitHub account
4. Select your repository
5. Configure:
   - **Name**: `stock-prediction-webapp`
   - **Region**: Frankfurt (or nearest)
   - **Branch**: `main`
   - **Build Command**: `pip install -r requirements.txt`
   - **Start Command**: `gunicorn webapp:app --bind 0.0.0.0:$PORT --workers 2 --timeout 120`
   - **Plan**: Free

### Step 3: Set Environment Variables
In Render dashboard → Environment:
```
DEEPSEEK_API_KEY=your_deepseek_api_key_here
PYTHON_VERSION=3.11.0
```

### Step 4: Deploy!
Click **Create Web Service** and wait ~5-10 minutes.

Your URL will be: `https://stock-prediction-webapp.onrender.com`

### Limitations
- Sleeps after 15 min inactivity (30s cold start)
- 750 free hours/month (~31 days)
- 512MB RAM

---

## Option 2: Railway.app

### Step 1: One-Click Deploy
1. Go to [railway.app](https://railway.app)
2. Click **New Project** → **Deploy from GitHub**
3. Select your repo
4. Railway auto-detects Python and deploys

### Step 2: Set Environment Variables
```
DEEPSEEK_API_KEY=your_key_here
```

### URL
`https://your-app.up.railway.app`

### Limitations
- $5 credit/month (~500 hours)
- Faster than Render (better cold starts)

---

## Option 3: Hugging Face Spaces (ML-Focused)

Best if you want to share as an ML demo.

### Step 1: Create Space
1. Go to [huggingface.co/spaces](https://huggingface.co/spaces)
2. Click **Create Space**
3. Choose **Gradio** or **Docker**

### Step 2: Upload Files
```
your-space/
├── app.py          # Gradio/Streamlit wrapper
├── webapp.py       # Your Flask app
├── requirements.txt
├── models/
└── src/
```

### Step 3: Create Gradio Wrapper
```python
# app.py
import gradio as gr
from webapp import generate_prediction

def predict(ticker):
    result = generate_prediction(ticker)
    return f"Signal: {result['signal']}\nConfidence: {result['confidence']}"

demo = gr.Interface(
    fn=predict,
    inputs=gr.Textbox(label="Ticker (e.g., AAPL, 0700.HK)"),
    outputs=gr.Textbox(label="Prediction"),
    title="Stock Prediction Model"
)

demo.launch()
```

---

## Option 4: Local Network Sharing (Quickest for Teammates)

If you just want teammates on the same network to test:

### Using ngrok (Free)
```bash
# Install ngrok
pip install pyngrok

# Run your app
python webapp.py

# In another terminal, expose it
ngrok http 5000
```

You'll get a public URL like: `https://abc123.ngrok.io`

### Using localtunnel
```bash
npx localtunnel --port 5000
```

---

## Important Files for Deployment

### Procfile (Required for Render/Railway)
```
web: gunicorn webapp:app --bind 0.0.0.0:$PORT --workers 2 --timeout 120
```

### render.yaml (Auto-config for Render)
```yaml
services:
  - type: web
    name: stock-prediction-webapp
    env: python
    plan: free
    buildCommand: pip install -r requirements.txt
    startCommand: gunicorn webapp:app --bind 0.0.0.0:$PORT --workers 2 --timeout 120
```

### runtime.txt (Python version)
```
python-3.11.0
```

---

## Troubleshooting

### "Application failed to start"
- Check logs in Render/Railway dashboard
- Ensure all imports work: `python -c "import webapp"`
- Verify requirements.txt has all dependencies

### "Module not found"
Add missing module to requirements.txt:
```bash
pip freeze > requirements.txt
```

### "Timeout during build"
- Free tier has limited build time
- Remove heavy dependencies (torch is ~2GB)
- Use lighter alternatives if possible

### "Out of memory"
- Free tier has 512MB RAM
- Reduce workers: `--workers 1`
- Disable FinBERT if needed (uses 400MB+)

---

## Lightweight Version (Without FinBERT)

If deployment fails due to memory/size, create a lightweight requirements:

```txt
# requirements-lite.txt
yfinance>=0.2.32
pandas>=2.0.0
numpy>=1.24.0
scikit-learn>=1.3.0
xgboost>=2.0.0
flask>=3.0.0
flask-cors>=4.0.0
gunicorn>=21.0.0
vaderSentiment>=3.3.2
ta>=0.11.0
```

The app will fall back to VADER sentiment (no FinBERT).

---

## Share with Teammates

Once deployed, share the URL:
```
https://stock-prediction-webapp.onrender.com

Endpoints:
- Homepage: /
- Predict: /api/predict/AAPL
- Top Picks: /api/top-picks
- Health: /api/health
```

---

*Last Updated: November 2024*
