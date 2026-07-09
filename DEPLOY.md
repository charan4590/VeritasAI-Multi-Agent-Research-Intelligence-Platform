# Deployment Guide

## Deploy to Render (Free — 5 minutes)

### Step 1 — Push to GitHub
```bash
git init
git add .
git commit -m "initial commit"
gh repo create research-agent --public --push
```
Or manually create a repo at github.com and push.

### Step 2 — Connect to Render
1. Go to https://render.com and sign up (free)
2. Click "New" → "Web Service"
3. Connect your GitHub repo
4. Render auto-detects `render.yaml` — click "Apply"

### Step 3 — Set environment variables in Render dashboard
Under "Environment" tab, add:
```
TAVILY_API_KEY=tvly-...        (from https://tavily.com)
GROQ_API_KEY=gsk_...           (from https://console.groq.com — FREE)
```

### Step 4 — Deploy
Click "Manual Deploy" → "Deploy latest commit"
Wait ~3 minutes → your app is live at:
`https://research-agent-xxxx.onrender.com`

---

## Getting Free API Keys

### Groq (replaces Ollama on server — fast + free)
1. https://console.groq.com
2. Sign up → API Keys → Create Key
3. Free tier: 14,400 requests/day on llama-3.1-8b-instant

### Tavily (web search)
1. https://tavily.com
2. Sign up → Dashboard shows your key
3. Free tier: 1,000 searches/month

---

## Local Development
```bash
cd backend
pip install -r requirements.txt
cp .env.example .env
# Fill in TAVILY_API_KEY and either GROQ_API_KEY or ensure Ollama is running
uvicorn main:app --reload --port 8001
```

## Run with Docker locally
```bash
docker build -t research-agent .
docker run -p 8000:8000 \
  -e TAVILY_API_KEY=your_key \
  -e GROQ_API_KEY=your_key \
  research-agent
```
