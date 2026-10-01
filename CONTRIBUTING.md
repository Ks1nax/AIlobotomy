# Contributing to AIlobotomy

Thanks for wanting to operate on more patients! 🧠🔪

## Ideas that are especially welcome

- **New surgical tools.** An operation is a Python function in `server.py` that edits `S.patient` and returns a one-line description, plus an entry in `OPS` in `static/index.html` that describes its fields. Register it in the `OPS` dict on the server and it gets undo, toggling and recipes for free.
- **Recipes.** Found a combination that produces something funny or interesting? Add a JSON file to `recipes/` (export it with the **Export recipe** button) and mention the model and a sample answer in the PR.
- **English UI.** All interface strings live in `static/index.html`; operation descriptions come from `server.py`.
- **Architecture support.** If a model fails to load or an operation says it can't find a block, open an issue with the model id.

## Running locally

```bash
./start.sh          # Linux / macOS
start.bat           # Windows
```

Set `AILOBOTOMY_NO_BROWSER=1` to stop the browser from opening and `AILOBOTOMY_PORT` to change the port.

## Pull requests

- Keep the UI in a single `index.html` with no build step.
- Test with at least one small model (e.g. `Qwen/Qwen2.5-0.5B-Instruct`) and make sure **Reset** brings the patient back.
- Screenshots or a GIF of the result make reviews much faster.
