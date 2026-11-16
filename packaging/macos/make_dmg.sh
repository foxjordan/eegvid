#!/bin/bash
# Build the lab's Mac installer (a .dmg) from the Intel app.
#   packaging/macos/make_dmg.sh            # builds the app first, then the dmg
#   packaging/macos/make_dmg.sh --no-build # package an existing dist/x86_64 build
# Output: dist/EEGVideoRecorder-<version>-mac.dmg
#
# Signing: with a "Developer ID Application" certificate in the keychain the app and
# dmg are signed with it; with a notarytool keychain profile (default name
# "eegvid-notary", or EEGVID_NOTARY_PROFILE) the dmg is also notarized and stapled,
# so it opens with a plain double-click on any Mac. EEGVID_SIGN_ID picks a
# certificate explicitly. Without a certificate the build stays unsigned (ad hoc).
#
# The Intel (x86_64) app is the one to ship: BioSemi's Mac library is Intel-only, and
# the same app runs on Intel Macs and (through Rosetta) on Apple Silicon Macs.
set -euo pipefail
cd "$(dirname "$0")/../.."
VERSION=$(sed -n 's/^version = "\(.*\)"/\1/p' pyproject.toml)
[[ "$VERSION" =~ ^[0-9]+(\.[0-9]+)*([-.][A-Za-z0-9]+)*$ ]] || { echo "Bad version '$VERSION'"; exit 1; }
APP="dist/x86_64/EEG Video Recorder.app"

if [[ "${1:-}" != "--no-build" ]]; then
  if [[ "$(uname -m)" == "arm64" ]]; then
    PY=(arch -x86_64 .venv-x86_64/bin/python)
  else
    PY=(.venv-x86_64/bin/python)
    [[ -x .venv-x86_64/bin/python ]] || PY=(python3)
  fi
  "${PY[@]}" -m PyInstaller packaging/eegvid.spec --noconfirm \
      --distpath dist/x86_64 --workpath build/x86_64
fi
[[ -d "$APP" ]] || { echo "No app at $APP"; exit 1; }
[[ -d "$APP/Contents/Resources/drivers/BioSemiRingBuffer64.framework" || \
   -d "$APP/Contents/Frameworks/drivers/BioSemiRingBuffer64.framework" ]] || \
   echo "WARNING: the BioSemi library is not in the app (simulator only)"
SIGN_ID="${EEGVID_SIGN_ID:-$(security find-identity -v -p codesigning |
           sed -n 's/.*"\(Developer ID Application: .*\)"/\1/p' | head -1)}"
PROFILE="${EEGVID_NOTARY_PROFILE:-eegvid-notary}"
if [[ -n "$SIGN_ID" ]]; then
  packaging/macos/sign_app.sh "$APP" "$SIGN_ID"
else
  echo "No Developer ID certificate: the app stays unsigned (users must click 'Open Anyway')."
  codesign --verify --deep --strict "$APP"
fi

STAGE=$(mktemp -d)
trap 'rm -rf "$STAGE"' EXIT
ditto "$APP" "$STAGE/EEG Video Recorder.app"
ln -s /Applications "$STAGE/Applications"
cp packaging/INSTALL.txt "$STAGE/Read Me First.txt"
OUT="dist/EEGVideoRecorder-$VERSION-mac.dmg"
rm -f "$OUT" "${OUT%.dmg}-UNSIGNED.dmg"
hdiutil create -volname "EEG Video Recorder $VERSION" -srcfolder "$STAGE" -ov -format UDZO "$OUT" >/dev/null
NOTARIZED=0
if [[ "$SIGN_ID" == Developer\ ID* ]]; then
  codesign --force --timestamp --sign "$SIGN_ID" "$OUT"
  if xcrun notarytool history --keychain-profile "$PROFILE" >/dev/null 2>&1; then
    echo "Notarizing (usually a few minutes)..."
    NOTA_JSON=$(mktemp)
    xcrun notarytool submit "$OUT" --keychain-profile "$PROFILE" --wait --output-format json > "$NOTA_JSON" ||
      { cat "$NOTA_JSON"; echo "notarytool failed"; exit 1; }
    STATUS=$(plutil -extract status raw -o - "$NOTA_JSON" 2>/dev/null || true)
    SUB_ID=$(plutil -extract id raw -o - "$NOTA_JSON" 2>/dev/null || true)
    rm -f "$NOTA_JSON"
    echo "Notarization status: ${STATUS:-unknown} (submission ${SUB_ID:-?})"
    if [[ "$STATUS" != "Accepted" ]]; then
      echo "Notarization was NOT accepted. Details: xcrun notarytool log $SUB_ID --keychain-profile $PROFILE"
      exit 1
    fi
    xcrun stapler staple "$OUT"
    spctl --assess --type open --context context:primary-signature -vv "$OUT"
    spctl --assess --type execute -vv "$APP"
    NOTARIZED=1
  else
    echo "No notarytool profile '$PROFILE': signed but NOT notarized."
  fi
fi
if [[ $NOTARIZED == 0 ]]; then
  # Never let a build macOS will warn about pass for the release.
  mv "$OUT" "${OUT%.dmg}-UNSIGNED.dmg"
  OUT="${OUT%.dmg}-UNSIGNED.dmg"
  echo "NOTE: not notarized; lab Macs will warn when opening $OUT"
fi
echo "Built $OUT ($(du -h "$OUT" | cut -f1))"
