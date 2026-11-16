#!/usr/bin/env bash
# usage: probe2.sh <version-dir> <outdir>  -- each scenario in its own CODEX_HOME
set -u
P=/tmp/claude-0/-home-user-daydream/c8f377bd-7126-54e9-978f-029f3676a70f/scratchpad/codex-probe
BIN=$P/$1/node_modules/.bin/codex
OUT=$2; rm -rf "$OUT"; mkdir -p "$OUT"
export NO_PROXY=127.0.0.1,localhost no_proxy=127.0.0.1,localhost
REPO=$OUT/repo; mkdir -p $REPO; git -C $REPO init -q; echo hi > $REPO/a.txt; git -C $REPO add a.txt; git -C $REPO -c user.email=x@y -c user.name=x commit -qm init
mkhome() { mkdir -p "$1"; printf 'model_provider = "fake"\n[model_providers.fake]\nname = "fake"\nbase_url = "http://127.0.0.1:18765/v1"\nwire_api = "responses"\nrequest_max_retries = 0\nstream_max_retries = 0\n' > "$1/config.toml"; }
run() { local name=$1 home=$2; shift 2
  ( cd $REPO && echo "probe prompt" | CODEX_HOME=$home timeout 60 "$BIN" exec --experimental-json --model fake-model --sandbox read-only --cd $REPO "$@" > $OUT/$name.stdout 2> $OUT/$name.stderr; echo $? > $OUT/$name.exit ); }
summ() { local name=$1 home=$2
  local rollouts=$(find $home/sessions -name 'rollout-*.jsonl' 2>/dev/null | wc -l)
  local lines=$(cat $(find $home/sessions -name 'rollout-*.jsonl' 2>/dev/null) /dev/null | wc -l)
  local threads=$(python3 -I -c 'import sqlite3,sys
try: print(sqlite3.connect(f"file:{sys.argv[1]}?mode=ro",uri=True).execute("select count(*) from threads").fetchone()[0])
except Exception as e: print("n/a")' $(ls $home/state_*.sqlite 2>/dev/null | head -1) )
  local extra=$(cd $home && ls -d sessions thread-writer-locks thread_history_*.sqlite shell_snapshots archived_sessions history.jsonl 2>/dev/null | tr '\n' ' ')
  local evts=$(python3 -I -c 'import json,sys
out=[]
for l in open(sys.argv[1]):
    try: e=json.loads(l)
    except Exception: continue
    t=e.get("type"); out.append(t if t!="turn.completed" else "turn.completed(in=%s)"%e["usage"]["input_tokens"])
print(",".join(out))' $OUT/$name.stdout)
  echo "$name exit=$(cat $OUT/$name.exit) rollouts=$rollouts rollout_lines=$lines threads_rows=$threads present=[$extra] events=$evts"; }
tid() { python3 -I -c 'import json,sys
for l in open(sys.argv[1]):
    try: e=json.loads(l)
    except Exception: continue
    if e.get("type")=="thread.started": print(e["thread_id"]); break' $1; }
mkhome $OUT/h_persist; run persist $OUT/h_persist; summ persist $OUT/h_persist
mkhome $OUT/h_eph; run eph $OUT/h_eph --ephemeral; summ eph $OUT/h_eph
for v in plain eph_before eph_after; do
  H=$OUT/h_$v; mkhome $H; run seed_$v $H; summ seed_$v $H; T=$(tid $OUT/seed_$v.stdout)
  case $v in plain) run resume_$v $H resume "$T";; eph_before) run resume_$v $H --ephemeral resume "$T";; eph_after) run resume_$v $H resume "$T" --ephemeral;; esac
  summ resume_$v $H; echo "  resumed_same_thread=$([ "$(tid $OUT/resume_$v.stdout)" = "$T" ] && echo yes || echo no)"
done
