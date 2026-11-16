#!/usr/bin/env bash
# usage: probe.sh <version-dir> <outdir>
set -u
P=/tmp/claude-0/-home-user-daydream/c8f377bd-7126-54e9-978f-029f3676a70f/scratchpad/codex-probe
BIN=$P/$1/node_modules/.bin/codex
OUT=$2; rm -rf "$OUT"; mkdir -p "$OUT"
export NO_PROXY=127.0.0.1,localhost no_proxy=127.0.0.1,localhost
REPO=$OUT/repo; mkdir -p $REPO; git -C $REPO init -q; echo hi > $REPO/a.txt; git -C $REPO add a.txt; git -C $REPO -c user.email=x@y -c user.name=x commit -qm init
mkhome() { mkdir -p "$1"; cat > "$1/config.toml" <<CFG
model_provider = "fake"
[model_providers.fake]
name = "fake"
base_url = "http://127.0.0.1:18765/v1"
wire_api = "responses"
request_max_retries = 0
stream_max_retries = 0
CFG
}
run() { # name home args...
  local name=$1 home=$2; shift 2
  ( cd $REPO && echo "probe prompt" | CODEX_HOME=$home timeout 60 "$BIN" exec --experimental-json --model fake-model --sandbox read-only --cd $REPO "$@" > $OUT/$name.stdout 2> $OUT/$name.stderr; echo $? > $OUT/$name.exit )
  ( cd $home && find . -type f ! -name config.toml | sort > $OUT/$name.files )
}
mkhome $OUT/home_persist; run persist $OUT/home_persist
mkhome $OUT/home_eph; run eph $OUT/home_eph --ephemeral
TID=$(python3 -I -c 'import json,sys
for l in open(sys.argv[1]):
    try: e=json.loads(l)
    except Exception: continue
    if e.get("type")=="thread.started": print(e["thread_id"]); break' $OUT/persist.stdout)
echo "$TID" > $OUT/tid
# persisted resume
cp -r $OUT/home_persist $OUT/home_resume
( cd $OUT/home_resume && find . -type f ! -name config.toml -exec sha256sum {} + | sort -k2 > $OUT/resume.before )
run resume $OUT/home_resume resume "$TID"
( cd $OUT/home_resume && find . -type f ! -name config.toml -exec sha256sum {} + | sort -k2 > $OUT/resume.after )
# ephemeral resume: flag before and after the subcommand
for pos in before after; do
  cp -r $OUT/home_persist $OUT/home_eresume_$pos
  ( cd $OUT/home_eresume_$pos && find . -type f ! -name config.toml -exec sha256sum {} + | sort -k2 > $OUT/eresume_$pos.before )
  if [ $pos = before ]; then run eresume_$pos $OUT/home_eresume_$pos --ephemeral resume "$TID"; else run eresume_$pos $OUT/home_eresume_$pos resume "$TID" --ephemeral; fi
  ( cd $OUT/home_eresume_$pos && find . -type f ! -name config.toml -exec sha256sum {} + | sort -k2 > $OUT/eresume_$pos.after )
done
