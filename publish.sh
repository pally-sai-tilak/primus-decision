#!/usr/bin/env bash
# Publish this tree as a NEW repository with a single-commit history (Primus Decision 0.1 — Research Alpha).
# Usage (from a clean copy of this directory, with the publisher's git identity configured):
#     ./publish.sh <empty-remote-url>          e.g. https://github.com/<owner>/primus-decision.git
# Repository name chosen by the owner: primus-decision (future releases live in the same repository); tag: primus-decision-0.1.
# It never touches any other repository or branch. The tree hash it prints is the content identity of the published
# commit; the release notes on the hosting site record the expected value (with Git LFS installed).
# The commit and tag are unsigned unless PUBLISH_SIGN=true, which uses the identity's configured signing key.
set -euo pipefail
REMOTE="${1:?usage: ./publish.sh <empty-remote-url>}"
[ -d .git ] && { echo "refusing: this directory already has a git history"; exit 1; }
command -v git-lfs >/dev/null 2>&1 || { echo "git-lfs is required for the two 71 MB featurizers"; exit 1; }
NAME="$(git config user.name || true)"
EMAIL="$(git config user.email || true)"
[ -n "$NAME" ] && [ -n "$EMAIL" ] || { echo "refusing: set git user.name and user.email to the publisher's identity first"; exit 1; }
case "$EMAIL" in *@anthropic.com) echo "refusing: the identity of the tool that prepared this tree must not author the public commit"; exit 1 ;; esac
if command -v sha256sum >/dev/null 2>&1; then sha256sum -c --quiet SHA256SUMS; else shasum -a 256 -c SHA256SUMS >/dev/null; fi || { echo "refusing: SHA256SUMS does not verify"; exit 1; }
EXTRA="$(find . -type f ! -path './.git/*' | sed 's|^\./||' | grep -v -x -F -e SHA256SUMS -f <(awk '{print $2}' SHA256SUMS) || true)"
[ -z "$EXTRA" ] || { echo "refusing: files not listed in SHA256SUMS:"; echo "$EXTRA"; exit 1; }
SIGN="${PUBLISH_SIGN:-false}"
echo "publishing as: $NAME <$EMAIL> (signed: $SIGN)"
git init -q -b main
git lfs install --local >/dev/null
git add -A
git -c commit.gpgsign="$SIGN" commit -q -m "Primus Decision 0.1 — Research Alpha"
echo "tree: $(git rev-parse HEAD^{tree})   (compare with the value in the release notes)"
git -c tag.gpgsign="$SIGN" tag -a primus-decision-0.1 -m "Primus Decision 0.1 — Research Alpha"
git remote add origin "$REMOTE"
git push -u origin main --tags
echo "published: $REMOTE  tag primus-decision-0.1"
