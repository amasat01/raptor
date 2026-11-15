#!/bin/sh
# The Rust legs of the gate. Usage: sh ci/gate_rs.sh [leg ...]   (no argument: every leg, in this order)
#   fmt clippy test headers exports globals nopython words
# Each leg prints what it examined and is RED when it examined nothing. Legs run one after the other (cargo at -j 2) and
# the exit status is non-zero when any leg is red. Needs cargo, clang and nm on PATH; no network.
cd "$(dirname "$0")/.." || exit 2
JOBS=${JOBS:-2}
HDRS="rs/ir/include/raptor_ir.h rs/ir/include/raptor_ir_query.h rs/wings/include/raptor_wings.h"
SRC_DIRS="rs/core rs/ir rs/wings"
out=$(mktemp "${TMPDIR:-/tmp}/gate_rs.XXXXXX")
trap 'rm -f "$out" "$out".*' EXIT

# RED when the number of things examined is zero
need() { [ "$2" -gt 0 ] || { echo "RED: $1 examined nothing"; return 1; }; }
# the .rs files tracked under the given directories (not tests unless $1 = all)
rs_files() { git ls-files $SRC_DIRS | grep '\.rs$' | grep -v "/tests/" ; }
# a summary of the cargo messages of a log
cargo_summary() { grep -E "^(warning|error)" "$out" | sort | uniq -c | sort -rn | head -8; }
# the passed count of every "test result" line of a log
passed() { awk '/^test result:/ { p += $4 } END { print p + 0 }' "$out"; }

leg_fmt() {
  cargo fmt --all --check >"$out" 2>&1 || { head -5 "$out"; echo "RED: fmt differs"; return 1; }
  n=$(git ls-files rs | grep -c '\.rs$'); echo "examined: $n rust files"; need fmt "$n"
}

leg_clippy() {
  cargo clippy --release -j "$JOBS" -p rcore -p raptor_ir -p raptor_wings -p raptor_support --all-targets -- -D warnings >"$out" 2>&1
  rc=$?; cargo_summary
  echo "examined: 4 packages (rcore raptor_ir raptor_wings raptor_support; the allows each crate carries stay)"
  [ $rc -eq 0 ] || { echo "RED: clippy"; return 1; }
}

leg_test() {
  cargo test --release -j "$JOBS" -p rcore -p raptor_ir -p raptor_wings -p raptor_support >"$out" 2>&1
  rc=$?; n=$(passed); grep -E "FAILED|panicked|^error" "$out" | head -5
  echo "examined: $n tests passed in 4 packages"
  [ $rc -eq 0 ] || { echo "RED: cargo test"; return 1; }; need test "$n"
}

leg_headers() {
  n=0
  for h in $HDRS; do
    clang -x c -fsyntax-only -Wall -Wextra -Werror -I rs/ir/include -I rs/wings/include "$h" || { echo "RED: $h"; return 1; }
    n=$((n + 1))
  done
  echo "examined: $n headers"; need headers "$n"
}

# every prototype of the headers has an exported symbol in the built libraries, and every exported rx_ / rw_ symbol a prototype
leg_exports() {
  cargo build --release -j "$JOBS" -p raptor_ir -p raptor_wings >"$out" 2>&1 || { grep -E "^error" "$out" | head -3; echo "RED: build"; return 1; }
  for h in $HDRS; do cat "$h"; done | tr '\n' ' ' | sed -E -e ':a' -e 's#/\*[^*]*\*+([^/*][^*]*\*+)*/##g' -e 'ta' |
    grep -oE '\b(rx|rw)_[a-z0-9_]+ *\(' | tr -d ' (' | sort -u >"$out.proto"
  # the release libraries carry link-time-optimisation bitcode that some `nm` cannot read: the first of $NM, llvm-nm, the homebrew
  # llvm-nm and nm that lists any symbol is used
  for nmtool in ${NM:-} llvm-nm "${HOMEBREW_PREFIX:-$HOME/homebrew}/opt/llvm/bin/llvm-nm" nm; do
    command -v "$nmtool" >/dev/null 2>&1 || continue
    for lib in target/release/libraptor_ir.a target/release/libraptor_wings.a; do
      "$nmtool" -g "$lib" 2>/dev/null | awk '$2 == "T" { print $3 }'
    done | sed 's/^_//' | grep -E '^(rx|rw)_' | sort -u >"$out.exp"
    [ -s "$out.exp" ] && break
  done
  np=$(wc -l <"$out.proto" | tr -d ' '); ne=$(wc -l <"$out.exp" | tr -d ' ')
  echo "examined: $np prototypes, $ne exported rx_/rw_ symbols"
  need exports "$np" && need exports "$ne" || return 1
  noexp=$(comm -23 "$out.proto" "$out.exp"); noproto=$(comm -13 "$out.proto" "$out.exp")
  [ -z "$noexp" ] || echo "prototype without an export: $(echo $noexp)"
  [ -z "$noproto" ] || echo "export without a prototype: $(echo $noproto)"
  [ -z "$noexp$noproto" ] || { echo "RED: header and exports differ"; return 1; }
}

leg_globals() {
  files=$(rs_files); n=$(echo "$files" | grep -c .)
  echo "examined: $n rust files in $SRC_DIRS"; need globals "$n" || return 1
  # shellcheck disable=SC2086
  hits=$(grep -nE 'static +mut|thread_local!|OnceLock|LazyLock|OnceCell|lazy_static|static +[A-Za-z_0-9]+ *: *[A-Za-z_:]*(Mutex|RwLock|Atomic[A-Za-z0-9]*)' $files)
  [ -z "$hits" ] || { echo "$hits" | head -10; echo "RED: global mutable state: $(echo "$hits" | wc -l | tr -d ' ') site(s)"; return 1; }
}

leg_nopython() {
  cargo tree -p raptor_ir -p raptor_wings -p rcore >"$out" 2>&1 || { head -3 "$out"; echo "RED: cargo tree"; return 1; }
  n=$(grep -c . "$out"); echo "examined: $n lines of the dependency tree"; need nopython "$n" || return 1
  hits=$(grep -iE 'pyo3|python' "$out" | head -5)
  [ -z "$hits" ] || { echo "$hits"; echo "RED: a python crate is in the compiled path"; return 1; }
}

# Counts of front-end and target words per file, against ci/rs_words_allowed.txt (file:word:count # reason): the list may only shrink.
words_now() {
  # shellcheck disable=SC2046
  awk '
    BEGIN { nw = split("hawk eagle Python repr pyo3 GRefMirror ScalarHandle", W, " "); np = split("aether:: std::size_t __fold_", P, " ") }
    { n = split($0, tok, /[^A-Za-z0-9_]+/); for (i = 1; i <= n; i++) for (j = 1; j <= nw; j++) if (tok[i] == W[j]) c[FILENAME ":" W[j]]++
      for (j = 1; j <= np; j++) { line = $0; while ((k = index(line, P[j])) > 0) { c[FILENAME ":" P[j]]++; line = substr(line, k + length(P[j])) } } }
    END { for (k in c) print k ":" c[k] }' $(rs_files) | sort
}
leg_words() {
  n=$(rs_files | grep -c .); echo "examined: $n rust files (not tests), 10 words; allow-list $(grep -c . ci/rs_words_allowed.txt) entries"
  need words "$n" || return 1
  words_now >"$out.now"
  sed 's/ *#.*//' ci/rs_words_allowed.txt | grep . | sort >"$out.allow"
  bad=$(awk -v allow="$out.allow" '
    function parse(r,   i, rest, j) { i = index(r, ":"); file = substr(r, 1, i - 1); rest = substr(r, i + 1)
      j = match(rest, /:[0-9]+$/); word = substr(rest, 1, j - 1); cnt = substr(rest, j + 1) + 0; key = file ":" word }
    FILENAME == allow { parse($0); a[key] = cnt; next }
    { parse($0); if (!(key in a)) print "new: " $0
      else if (cnt > a[key]) print "above the list (" a[key] "): " $0
      else if (cnt < a[key]) print "below the list (" a[key] "), shrink the list: " $0
      seen[key] = 1 }
    END { for (k in a) if (!(k in seen)) print "gone, shrink the list: " k ":" a[k] }' "$out.allow" "$out.now")
  [ -z "$bad" ] || { echo "$bad" | head -15; echo "RED: words differ from ci/rs_words_allowed.txt"; return 1; }
}

legs=${*:-fmt clippy test headers exports globals nopython words}
green=; red=
for l in $legs; do
  echo "== $l"
  if "leg_$l"; then green="$green $l"; else red="$red $l"; fi
done
echo "GREEN:$green"; echo "RED:$red"
[ -z "$red" ]
