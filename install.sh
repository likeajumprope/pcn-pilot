#!/usr/bin/env bash
# Install the three PCN-Pilot skills where an agent can find them.
#
#   ./install.sh                 # personal skills: ~/.claude/skills
#   ./install.sh /path/to/dir    # any other skills directory (e.g. <repo>/.claude/skills)
#   ./install.sh --link [dir]    # symlink instead of copy (for development)
#
# Existing folders with the same names are replaced by this version.
set -euo pipefail
here="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
mode=copy
if [[ "${1:-}" == "--link" ]]; then mode=link; shift; fi
dest="${1:-$HOME/.claude/skills}"
python3 "$here/tools/sync_common.py" >/dev/null
mkdir -p "$dest"
for skill in pcn-data-skill pcn-model-skill pcn-qc-skill; do
  rm -rf "${dest:?}/$skill"
  if [[ $mode == link ]]; then ln -s "$here/skills/$skill" "$dest/$skill"; else cp -R "$here/skills/$skill" "$dest/$skill"; fi
  echo "installed $skill -> $dest/$skill"
done
echo
echo "Next: copy skills/pcn-model-skill/assets/pipeline.env.template to ~/.config/pcnpilot/pipeline.env,"
echo "fill it in, then run:  \$PCN_PYTHON $here/selftest/run_selftest.py"
