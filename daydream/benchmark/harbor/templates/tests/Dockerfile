FROM python:3.12-slim@sha256:876416ecde9aca2bcc90e1fb0c7a9500bbf749f5788b70f82d4c5a5c2357f8b4

# Separate verifier context: stdlib plus the runtime-pinned HTTP client.
RUN pip install --no-cache-dir "httpx==0.28.1"

COPY test.sh score_review.py verifier_core.py judge_prompt.md golden-review.json ./
RUN chmod +x test.sh
