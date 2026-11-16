#!/bin/bash
# Sign the app for distribution: Developer ID + hardened runtime + secure timestamp,
# every Mach-O inside first, then the bundle. Usage: sign_app.sh APP "IDENTITY"
set -euo pipefail
APP="$1"; ID="$2"
ENT="$(cd "$(dirname "$0")" && pwd)/entitlements.plist"
MAIN="$APP/Contents/MacOS/EEGVideoRecorder"
TS=(--timestamp)
[[ "$ID" == "-" || "$ID" == Apple\ Development* ]] && TS=(--timestamp=none)  # test signing only

# Inside-out: the deepest paths first, so nested code is signed before what contains it.
find "$APP/Contents" -type f ! -path "$MAIN" |
  while IFS= read -r f; do if file -b "$f" | grep -q "Mach-O"; then
    printf '%d\t%s\n' "$(tr -cd / <<<"$f" | wc -c)" "$f"; fi; done |
  sort -t$'\t' -k1,1rn | cut -f2- | tr '\n' '\0' |
  xargs -0 -n 40 codesign --force --options runtime "${TS[@]}" --sign "$ID"
codesign --force --options runtime "${TS[@]}" --entitlements "$ENT" --sign "$ID" "$MAIN"
codesign --force --options runtime "${TS[@]}" --entitlements "$ENT" --sign "$ID" "$APP"
codesign --verify --deep --strict "$APP"
echo "Signed $APP with: $ID"
