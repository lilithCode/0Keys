#!/usr/bin/env sh
set -e
DIR="$(cd "$(dirname "$0")" && pwd)"
cd "$DIR"
AUTOSTART=0
[ "$1" = "--autostart" ] && AUTOSTART=1

PYTHON="${PYTHON:-}"
if [ -z "$PYTHON" ]; then
    for candidate in python3.12 python3.11 python3.10 python3; do
        if command -v "$candidate" >/dev/null 2>&1; then PYTHON="$candidate"; break; fi
    done
fi
if [ -z "$PYTHON" ]; then
    echo "Python 3.10-3.12 is needed. Install it, then run this script again." >&2
    exit 1
fi
if ! "$PYTHON" -c 'import sys, tkinter; sys.exit(not (3, 10) <= sys.version_info[:2] <= (3, 12))' 2>/dev/null; then
    echo "$PYTHON is not Python 3.10-3.12 with Tk. Run with PYTHON=/path/to/python3.12 ./install.sh" >&2
    exit 1
fi

echo "==> Python packages"
[ -x .venv/bin/python ] || "$PYTHON" -m venv .venv
.venv/bin/python -m pip install --quiet --upgrade pip
.venv/bin/python -m pip install --quiet -r requirements.txt

echo "==> Hand tracking model"
mkdir -p models config logs
if [ ! -f models/hand_landmarker.task ]; then
    .venv/bin/python - <<'EOF'
import urllib.request
url = ("https://storage.googleapis.com/mediapipe-models/hand_landmarker/"
       "hand_landmarker/float16/1/hand_landmarker.task")
urllib.request.urlretrieve(url, "models/hand_landmarker.task.part")
import os
os.replace("models/hand_landmarker.task.part", "models/hand_landmarker.task")
EOF
fi
chmod +x 0keys

echo "==> Typing into other apps"
.venv/bin/python -c 'from helper_output import choose_sender; print("   ", choose_sender()[1])'

if [ "$(uname)" = "Linux" ]; then
    APPS="${XDG_DATA_HOME:-$HOME/.local/share}/applications"
    mkdir -p "$APPS"
    cat > "$APPS/0keys.desktop" <<EOF
[Desktop Entry]
Type=Application
Name=0Keys
Comment=Type into any app by tapping the table
Exec="$DIR/0keys"
Terminal=false
Categories=Utility;Accessibility;
EOF
    echo "==> Added 0Keys to the application menu"
    if [ "$AUTOSTART" = 1 ]; then
        AUTO="${XDG_CONFIG_HOME:-$HOME/.config}/autostart"
        mkdir -p "$AUTO"
        sed 's|^Exec=.*|Exec="'"$DIR"'/0keys" --minimized|' "$APPS/0keys.desktop" > "$AUTO/0keys.desktop"
        echo "==> 0Keys will start when you log in"
        if [ -n "$HYPRLAND_INSTANCE_SIGNATURE" ]; then
            echo "    Hyprland does not read autostart files; add to hyprland.conf:"
            echo "    exec-once = \"$DIR/0keys\" --minimized"
        fi
    fi
fi

echo
echo "Done. Start it with: $DIR/0keys"
echo "First click 'Set up keyboard', then press Ctrl+Alt+K in any app to type."
if [ "$(uname)" = "Darwin" ]; then
    echo "macOS: allow your terminal or Python under System Settings > Privacy & Security >"
    echo "Accessibility and Input Monitoring, so the hotkey and typing work."
fi
