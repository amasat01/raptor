#!/usr/bin/env bash
# Copyright 2026 Alessandro Masat
# SPDX-License-Identifier: Apache-2.0
#
# Set up a development machine for the RAPTOR family: clone aether, raptor,
# eagle and hawk, build the two pinned environments (gpu and cpu profiles, from
# eagle/benchmarks/reproduce/locks/) with editable installs and compiled _cores,
# run each repository's quick gate, and write an env script. Idempotent: re-run
# it after a pull to rebuild what changed. Nothing secret is written to disk;
# private repositories are cloned over SSH through your running SSH agent.
#
#   setup_devbox.sh [options]
#     --root DIR          where the checkouts live (default ~/raptor)
#     --work DIR          envs, build trees, env scripts (default ROOT/.devbox)
#     --ref REPO=REF      branch/tag to clone for a family repo (default main)
#     --extra-repo N=URL  also clone N from URL (e.g. an SSH URL) and install it
#                         editable into both envs (repeatable)
#     --profiles LIST     "gpu cpu" (default), or one of them
#     --arch LIST         CUDA archs for the gpu env (default: the visible GPUs',
#                         else eagle's default list)
#     --jobs N            parallel build jobs (default: all cores)
#     --no-gates          skip the quick gates
#     --cxx-gates         also build and run aether's and eagle's C++ suites (CPU mode)
#     --dry-run           print every command, change nothing
set -euo pipefail

ROOT=$HOME/raptor WORK="" PROFILES="gpu cpu" ARCH="" GATES=1 CXX_GATES=0 DRY=0
JOBS=$(nproc 2>/dev/null || echo 4)
GIT_BASE=${REPRO_GIT_BASE:-https://github.com/amasat01}
REPOS=(aether raptor eagle hawk)
declare -A REF=([aether]=main [raptor]=main [eagle]=main [hawk]=main)
declare -A EXTRA=()

log() { printf '[devbox] %s\n' "$*" >&2; }
die() { printf '[devbox] ERROR: %s\n' "$*" >&2; exit 1; }
run() { printf '+ %s\n' "$*"; [[ $DRY == 1 ]] || "$@"; }

while [[ $# -gt 0 ]]; do
  case $1 in
    --root) ROOT=$2; shift 2 ;;
    --work) WORK=$2; shift 2 ;;
    --ref) [[ $2 == *=* && -n ${REF[${2%%=*}]+x} ]] || die "--ref wants REPO=REF, REPO in ${REPOS[*]}"
           REF[${2%%=*}]=${2#*=}; shift 2 ;;
    --extra-repo) [[ $2 == *=* ]] || die "--extra-repo wants NAME=URL"; EXTRA[${2%%=*}]=${2#*=}; shift 2 ;;
    --profiles) PROFILES=$2; shift 2 ;;
    --arch) ARCH=$2; shift 2 ;;
    --jobs) JOBS=$2; shift 2 ;;
    --no-gates) GATES=0; shift ;;
    --cxx-gates) CXX_GATES=1; shift ;;
    --dry-run) DRY=1; shift ;;
    -h|--help) sed -n '5,26p' "$0"; exit 0 ;;
    *) die "unknown option $1 (see --help)" ;;
  esac
done
WORK=${WORK:-$ROOT/.devbox}

# --- machine check (advisory) ---------------------------------------------------
cores=$(nproc); mem=$(awk '/MemTotal/ {printf "%d", $2/1048576}' /proc/meminfo)
run mkdir -p "$ROOT" "$WORK"
d=$ROOT; while [[ ! -d $d ]]; do d=$(dirname "$d"); done
free=$(df -BG --output=avail "$d" 2>/dev/null | tail -1 | tr -dc 0-9 || true); free=${free:-0}
log "machine: $cores cores, ${mem} GiB RAM, ${free} GiB free under $ROOT"
(( cores >= 8 )) || log "warning: fewer than 8 cores; builds and gates will be slow"
(( mem >= 60 )) || log "warning: less than 64 GiB RAM; keep --jobs low"
(( free >= 100 )) || log "warning: less than 100 GiB free; two envs + build trees need ~60 GiB"

# --- 1. checkouts ---------------------------------------------------------------
for r in "${REPOS[@]}"; do
  if [[ -d $ROOT/$r/.git ]]; then log "$r: existing checkout kept as it is"
  else run git clone -q --branch "${REF[$r]}" "$GIT_BASE/$r.git" "$ROOT/$r"; fi
done
for n in "${!EXTRA[@]}"; do
  if [[ -d $ROOT/$n/.git ]]; then log "$n: existing checkout kept as it is"
  elif GIT_TERMINAL_PROMPT=0 git ls-remote -q "${EXTRA[$n]}" HEAD >/dev/null 2>&1 || [[ $DRY == 1 ]]; then
    run git clone -q "${EXTRA[$n]}" "$ROOT/$n"
  else
    log "$n: ${EXTRA[$n]} not reachable with the current credentials; skipped"; unset "EXTRA[$n]"
  fi
done

# --- 2. environments (eagle's bootstrap, editable, against these checkouts) ----
BOOT=$ROOT/eagle/benchmarks/reproduce/bootstrap.sh
[[ -f $BOOT || $DRY == 1 ]] || die "$BOOT not found (eagle checkout too old?)"
DRYARG=(); [[ $DRY == 1 ]] && DRYARG=(--dry-run)
have_gpu=0; command -v nvidia-smi >/dev/null 2>&1 && nvidia-smi -L >/dev/null 2>&1 && have_gpu=1
for prof in $PROFILES; do
  args=(--target devbox --profile "$prof" --work "$WORK" --src "$ROOT" --editable --jobs "$JOBS")
  if [[ $prof == gpu ]]; then
    if [[ -n $ARCH ]]; then args+=(--arch "$ARCH")
    elif [[ $have_gpu == 0 ]]; then args+=(--arch "61-real;70-real;80-real;90-real;61-virtual"); fi
  fi
  run bash "$BOOT" "${args[@]}" "${DRYARG[@]}"
  for n in "${!EXTRA[@]}"; do
    run bash -c ". '$WORK/env-$prof.sh' && export CMAKE_PREFIX_PATH=\$CONDA_PREFIX && \
python -m pip install -q --no-deps --no-build-isolation -e '$ROOT/$n'"
  done
done

# --- 3. env script ----------------------------------------------------------------
ENVSH=$WORK/raptor_env.sh
if [[ $DRY == 1 ]]; then log "would write $ENVSH"; else
  cat >"$ENVSH" <<EOF
# RAPTOR devbox environment. Usage: . $ENVSH [gpu|cpu]   (default: gpu if built, else cpu)
_p=\${1:-\$( [ -f "$WORK/env-gpu.sh" ] && echo gpu || echo cpu)}
. "$WORK/env-\$_p.sh"; unset _p
export RAPTOR_ROOT=$ROOT
EOF
  log "wrote $ENVSH"
fi

# --- 4. quick gates ---------------------------------------------------------------
FAILED=()
gate() {  # gate NAME PROFILE DIR CMD...
  local name=$1 prof=$2 dir=$3; shift 3
  log "gate $name ($prof)"
  printf '+ (. %s; cd %s; %s)\n' "$WORK/env-$prof.sh" "$dir" "$*"
  [[ $DRY == 1 ]] && return 0
  # shellcheck disable=SC1090
  if ! ( . "$WORK/env-$prof.sh" && cd "$dir" && "$@" ); then FAILED+=("$name/$prof"); fi
}
if [[ $GATES == 1 ]]; then
  for prof in $PROFILES; do
    bt=$WORK/pytest/$prof; run mkdir -p "$bt"
    gate raptor "$prof" "$ROOT/raptor" python -m pytest -q -m "not cross_repo" --basetemp "$bt/raptor"
    gate aether-dsc "$prof" "$ROOT/aether/dsc" python -m pytest -q --basetemp "$bt/aether"
    if [[ $prof == gpu && $have_gpu == 1 ]]; then
      gate eagle "$prof" "$ROOT/eagle" python -m pytest -q python/tests -m "not interop_matrix" --basetemp "$bt/eagle"
      gate hawk "$prof" "$ROOT/hawk" python -m pytest -q tests --basetemp "$bt/hawk"
    else
      gate eagle "$prof" "$ROOT/eagle" python -m pytest -q python/tests -m "not gpu and not interop_matrix" --basetemp "$bt/eagle"
      gate hawk "$prof" "$ROOT/hawk" python -m pytest -q tests -m "not gpu" --basetemp "$bt/hawk"
    fi
  done
fi
if [[ $CXX_GATES == 1 ]]; then
  prof=${PROFILES##* }
  for r in aether eagle; do
    up=${r^^}; b=$WORK/build/cxx-$r
    gate "$r-cxx" "$prof" "$ROOT/$r" bash -c "cmake -S . -B '$b' -D${up}_CPP_MODE=ON -D${up}_BUILD_TESTS=ON \
-DCMAKE_PREFIX_PATH=\$CONDA_PREFIX/cpp\;\$CONDA_PREFIX -DCMAKE_INSTALL_PREFIX=\$CONDA_PREFIX/cpp && \
cmake --build '$b' -j $JOBS && { [ $r != aether ] || cmake --install '$b'; } && tests/check_gate.sh cpp '$b/tests/${r}_tests'"
  done
fi

if [[ ${#FAILED[@]} -gt 0 ]]; then
  log "FAILED gates: ${FAILED[*]}"; exit 1
fi
log "done. Use: . $ENVSH gpu   (or cpu)"
