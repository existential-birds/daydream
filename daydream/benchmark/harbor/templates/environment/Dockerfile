FROM __BASE_IMAGE__

RUN apt-get update \
    && apt-get install -y --no-install-recommends git \
    && rm -rf /var/lib/apt/lists/*

COPY runtime-requirements.lock /tmp/runtime-requirements.lock
RUN python -m pip install --no-cache-dir --require-hashes -r /tmp/runtime-requirements.lock \
    && rm /tmp/runtime-requirements.lock
COPY daydream-__DAYDREAM_VERSION__-py3-none-any.whl /tmp/daydream-__DAYDREAM_VERSION__-py3-none-any.whl
RUN python -m pip install --no-cache-dir --no-deps /tmp/daydream-__DAYDREAM_VERSION__-py3-none-any.whl \
    && rm /tmp/daydream-__DAYDREAM_VERSION__-py3-none-any.whl

COPY repository.bundle /tmp/repository.bundle
RUN git clone --no-checkout /tmp/repository.bundle /workspace/repo \
    && git -C /workspace/repo branch base origin/base \
    && git -C /workspace/repo branch head origin/head \
    && git -C /workspace/repo rev-parse --verify base \
    && git -C /workspace/repo rev-parse --verify head \
    && git -C /workspace/repo checkout head \
    && git -C /workspace/repo remote remove origin \
    && rm /tmp/repository.bundle

WORKDIR /workspace/repo
