FROM python:3.12-slim

# Self-contained verifier image: the generated judge environment needs only the
# stdlib plus httpx (no daydream wheel — the daydream agent env is separate).
RUN pip install --no-cache-dir "httpx>=0.28"

WORKDIR /verifier

COPY score_review.py verifier_core.py judge_prompt.md golden-review.json test.sh ./
RUN chmod +x test.sh

ENTRYPOINT ["python", "score_review.py"]