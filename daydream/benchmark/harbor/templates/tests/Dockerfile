FROM __BASE_IMAGE__

# Separate verifier environment: stdlib plus the hash-locked HTTP client.
# httpx and its transitive subtree are pinned version+content-hash with the
# committed closure (same as runtime-requirements.lock) and installed with
# --require-hashes, so the verifier -- the container carrying the judge API
# key to api.anthropic.com -- builds deterministically/digest-locked.
RUN printf 'httpx==0.28.1 --hash=sha256:75e98c5f16b0f35b567856f597f06ff2270a374470a5c2392242528e3e3e42fc --hash=sha256:d909fcccc110f8c7faf814ca82a9a4d816bc5a6dbfea25d6591d6985b8ba59ad\nhttpcore==1.0.9 --hash=sha256:2d400746a40668fc9dec9810239072b40b4484b640a8c38fd654a024c7a1bf55 --hash=sha256:6e34463af53fd2ab5d807f399a9b45ea31c3dfa2276f15a2c3f00afff6e176e8\nanyio==4.12.1 --hash=sha256:41cfcc3a4c85d3f05c932da7c26d0201ac36f72abd4435ba90d0464a3ffed703 --hash=sha256:d405828884fc140aa80a3c667b8beed277f1dfedec42ba031bd6ac3db606ab6c\ncertifi==2026.1.4 --hash=sha256:9943707519e4add1115f44c2bc244f782c0249876bf51b6599fee1ffbedd685c --hash=sha256:ac726dd470482006e014ad384921ed6438c457018f4b3d204aea4281258b2120\nh11==0.16.0 --hash=sha256:4e35b956cf45792e4caa5885e69fba00bdbc6ffafbfa020300e549b208ee5ff1 --hash=sha256:63cf8bbe7522de3bf65932fda1d9c2772064ffb3dae62d55932da54b31cb6c86\nidna==3.15 --hash=sha256:048adeaf8c2d788c40fee287673ccaa74c24ffd8dcf09ffa555a2fbb59f10ac8 --hash=sha256:ca962446ea538f7092a95e057da437618e886f4d349216d2b1e294abfdb65fdc\ntyping_extensions==4.15.0 --hash=sha256:0cea48d173cc12fa28ecabc3b837ea3cf6f38c6d1136f85cbaaf598984861466 --hash=sha256:f0fa19c6845758ab08074a0cfa8b7aecb71c999ca73d62883bc25cc018c4e548\n' > /tmp/httpx-lock.txt \
    && python -m pip install --no-cache-dir --require-hashes -r /tmp/httpx-lock.txt \
    && rm /tmp/httpx-lock.txt

COPY test.sh score_review.py verifier_core.py judge_prompt.md golden-review.json ./
RUN chmod +x test.sh
