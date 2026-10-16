#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
THIRD_PARTY="$ROOT/third_party"
SURGICAL_REV="916dcac6fb1c5fcccc9581471370645763c65ce6"
ADASTEER_REV="2a02b78bea83df497dd3495e7a0dda4580f2dd53"

fetch_pinned_repo() {
    local name="$1" url="$2" revision="$3"
    local path="$THIRD_PARTY/$name"
    if [[ -f "$path/UPSTREAM_COMMIT" ]] && [[ "$(cat "$path/UPSTREAM_COMMIT")" == "$revision" ]]; then
        echo "$name is already staged at $revision"
        return
    fi
    if [[ -e "$path" ]]; then
        echo "$path exists without the expected revision marker; inspect it before replacing it" >&2
        return 1
    fi
    git clone --filter=blob:none --no-checkout "$url" "$path"
    git -C "$path" fetch --depth 1 origin "$revision"
    git -C "$path" checkout --detach "$revision"
    printf '%s\n' "$revision" > "$path/UPSTREAM_COMMIT"
}

mkdir -p "$THIRD_PARTY"
fetch_pinned_repo False-Refusal-Mitigation https://github.com/mainlp/False-Refusal-Mitigation.git "$SURGICAL_REV"
fetch_pinned_repo AdaSteer https://github.com/MuyuenLP/AdaSteer.git "$ADASTEER_REV"

python - "$THIRD_PARTY/AdaSteer/adasteer/src/main_generate_steering_multi_adasteer.py" <<'PY'
from pathlib import Path
import sys
path = Path(sys.argv[1])
source = path.read_text()
old = "use_flash_attention_2 = True"
if old in source:
    source = source.replace(old, 'attn_implementation="sdpa"')
elif 'attn_implementation="sdpa"' not in source:
    raise SystemExit("Expected AdaSteer attention option was not found")
path.write_text(source)
PY

cat > "$THIRD_PARTY/AdaSteer/LOCAL_PATCHES.md" <<EOF
Upstream commit: $ADASTEER_REV

Compatibility patch: replace \`use_flash_attention_2=True\` with
\`attn_implementation="sdpa"\` in the Llama, Gemma, and Qwen model loaders.
Only Llama 3.1 is used in this project.
EOF
