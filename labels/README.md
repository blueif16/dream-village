# Context-card labels

1. **Open:** from the repo root run `.venv/bin/python -m http.server 8000` and open http://localhost:8000/labels/. The page loads `runs/scorer/full13-v1-v0-evalv2/cards/cards.jsonl`. For another run, add `?cards=../runs/<run>/cards/cards.jsonl`. If you open the file from disk, the page asks you to pick `cards.jsonl`. Don't look at `cards_key.jsonl` until you've finished labeling.
2. **Label:** read the question at the top. Press **Y** if the check passes, **N** if it fails, **S** if you're unsure (unsure cards don't count), and **←** / **→** to move between cards. Labels save automatically in this browser, so keep using the same URL.
3. **Export:** click "Export labels" to download `labels.jsonl` (one `{card_id, label, ts}` per line). Every export contains all your labels, so you can export again at any time.
4. **Agreement:** run `.venv/bin/python -m scorer.label_agreement ~/Downloads/labels.jsonl runs/scorer/full13-v1-v0-evalv2/cards/cards_key.jsonl`. It prints n, TPR, TNR and κ for each card type and judge. A judge is accepted when TPR ≥ 0.9, TNR ≥ 0.9 and κ ≥ 0.8 (design §7).
