# CoRAG Demo (Streamlit)

A side-by-side demo for Traditional RAG vs CoRAG-style iterative retrieval.

The app now supports a compare mode that runs both approaches side by side, auto-indexes `docs/` when needed, and shows a live step-by-step CoRAG trace.

## 1) Prerequisites

- Windows with WSL (Ubuntu) installed
- Python 3.10+
- A valid API key in `.env`:
  - `GOOGLE_API_KEY=...` or
  - `OPENAI_API_KEY=...`

## 2) Recommended run path (WSL home)

Use this path for better stability than `/mnt/c`.

Step A - Sync project into WSL home (one-time or when code changes):

wsl bash -lc "rm -rf ~/corag-demo; mkdir -p ~/corag-demo; rsync -av --exclude '.venv' --exclude '.venv-win' --exclude '**pycache**' --exclude 'chroma_db' /mnt/c/Users/Admin/Desktop/corag-demo/ ~/corag-demo/"

Step B - Create venv and install dependencies:

wsl bash -lc "cd ~/corag-demo; python3 -m venv .venv; source .venv/bin/activate; pip install --upgrade pip; pip install -r requirements.txt"

Step C - Run app:

wsl bash -lc "cd ~/corag-demo; source .venv/bin/activate; streamlit run app.py --server.address 0.0.0.0 --server.port 8501 --server.headless true"

Open in browser:

- http://localhost:8501
- If localhost is unstable, use WSL IP shown in terminal, for example `http://172.x.x.x:8501`.

## 3) Quick run directly from /mnt/c (alternative)

wsl bash -lc "cd /mnt/c/Users/Admin/Desktop/corag-demo; source .venv/bin/activate; streamlit run app.py --server.address 0.0.0.0 --server.port 8501 --server.headless true"

## 4) Stop stale Streamlit processes

If port 8501 is busy or app hangs:

wsl bash -lc "pkill -9 -f 'streamlit run app.py' || true; pkill -9 -f 'python3 -m streamlit' || true"

## 5) Demo flow

1. Start app.
2. Click one of the sample question buttons.
3. Click `Chay Demo`.
4. If no documents are indexed yet, app auto-indexes `docs/` before running.
5. Use `Compare both` when you want a direct side-by-side comparison.
6. CoRAG shows live retrieval steps and a short summary above the answer.

### What the comparison shows

- RAG: one retrieval pass, then a single answer.
- CoRAG: iterative retrieval with up to `max_steps`, then a final answer from the accumulated context.
- Metrics: steps, retrieved docs, latency, and simple answer coverage scores.

## 6) Optional health check

wsl bash -lc "curl -sS -m 5 http://127.0.0.1:8501/_stcore/health"

Expected output:

ok

## 7) Common issues

- Missing API key:
  - Ensure `.env` contains `GOOGLE_API_KEY` or `OPENAI_API_KEY`.
- Port already in use:
  - Run the stop command in section 4, then start again.
- Localhost cannot open:
  - Use WSL Network URL from Streamlit output.
- No indexed docs warning:
  - Ensure `docs/` has at least one `.txt` or `.pdf`, then click `Chay Demo` again.

## 8) Notes for the current demo corpus

- The `docs/` folder is intentionally structured for multi-hop retrieval.
- Some facts are repeated across files on purpose so CoRAG can chain retrieval across hops.
- If you update documents, rerun the app or click `Chay Demo` again so the vector store is rebuilt from the latest corpus.
