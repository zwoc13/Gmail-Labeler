#!/usr/bin/env bash
# Cleans up Gmail: Netflix, GitLab, Horoshop, Paperless Post, Bolt Ukraine, LinkedIn

set -euo pipefail

ACCOUNT="${1:-gmail}"
DRY_RUN="${2:-run}"  # pass "dry" as 2nd arg to preview only

declare -A CATEGORIES=(
  ["Netflix"]="from netflix.com"
  ["GitLab"]="from gitlab.com"
  ["Horoshop"]="from horoshop.com"
  ["Paperless Post"]="from paperlesspost.com"
  ["Bolt Ukraine"]="from bolt.ua"
  ["LinkedIn"]="from linkedin.com"
)

for category in "${!CATEGORIES[@]}"; do
  query="${CATEGORIES[$category]}"
  echo "[$category] Searching..."

  # Get message IDs
  ids=$(himalaya envelope list --account "$ACCOUNT" "$query" 2>/dev/null | \
        tail -n +4 | grep -E '^\|' | cut -d'|' -f2 | tr -d ' ' | sort -n | tr '\n' ' ')

  count=$(echo "$ids" | wc -w)
  echo "  Found $count emails: $ids"

  if [[ "$DRY_RUN" != "dry" ]] && [[ -n "$ids" ]]; then
    himalaya message delete --account "$ACCOUNT" $ids 2>/dev/null
    echo "  Deleted."
  else
    echo "  (skipped in dry run)"
  fi
done

echo "Done."
