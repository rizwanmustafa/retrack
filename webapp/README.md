# Retrack web application

Flask + MongoDB + Bootstrap web application for the Retrack multi-camera pedestrian detection and tracking
system, built to the Retrack SRS (`../Retrack_SRS.docx.pdf`).

* [SRS_CONTRADICTIONS.md](SRS_CONTRADICTIONS.md): where the SRS contradicts itself, and what was implemented.
* [SRS_ASSUMPTIONS.md](SRS_ASSUMPTIONS.md): every choice made where the SRS leaves room.
* [docs/TRACEABILITY.md](docs/TRACEABILITY.md): each FR, NFR and use case mapped to code and tests.

## What it does

| End User | Admin |
|---|---|
| Register (Pending until approved), sign in/out | Approve or reject registrations |
| Upload or select a calibration bundle (validated) | Approve or reject profile-update requests |
| Start LIVE or RECORDED monitoring, watch camera views with track IDs and the bird's eye view, Stop | Select the active model checkpoint (compatibility-checked) |
| Session history and summaries | Audit log and system errors |
| Generate, search and download PDF/CSV reports | Search, generate and download all reports |
| Request profile changes and edit approved fields | |

The model is reached only through the `AIModel` interface (`retrack_web/ai/`). The `tracktacular` backend wraps
`../WorldTrack` (TrackTacular + the Retrack tracker).

## Layout

```
retrack_web/
  __init__.py          application factory
  ai/                  AIModel interface + TrackTacular backend (the only code that imports WorldTrack)
  monitoring/          FrameSet contract, LIVE/RECORDED feeds, session engine (worker thread)
  reports/             summary figures, PDF/CSV rendering, generation/search/download
  services/            accounts, profile requests, calibrations, checkpoints
  blueprints/          routes: auth, enduser, admin, reports, api
  templates/, static/  Bootstrap 5.3 pages (Dashboard, Sign-in, Checkout and color-mode examples)
scripts/               calibration bundle builder, camera-rig client, session measurement
docs/                  formats, operations, traceability
tests/                 web-system tests (mongomock + fake model)
```

## Setup

Requirements: Python 3.10+, MongoDB 6+ (local, Docker or Atlas).

```bash
cd webapp
python3.11 -m venv .venv && source .venv/bin/activate
pip install -r requirements-dev.txt          # web app + tests
cp .env.example .env                         # then edit; export the variables or use a dotenv loader
docker run -d --name retrack-mongo -p 27017:27017 -v retrack-mongo:/data/db mongo:7   # if you have no MongoDB
```

For inference on the GPU host also install the model environment (CUDA build of torch/torchvision first):

```bash
pip install -r requirements-model.txt
```

Without it the application still runs; selecting a checkpoint or starting a session reports that the inference
environment is not installed.

## Run

```bash
export $(grep -v '^#' .env | xargs)
flask --app wsgi create-admin                 # provision the first Admin
flask --app wsgi run --port 5000              # development server
```

Then:

1. Put trained checkpoints (`last.ckpt` from the Retrack pipeline) in `RETRACK_CHECKPOINT_DIR`. As Admin open
   **Model checkpoints** and select one.
2. For RECORDED input create `instance/recorded_sources.json` (see `docs/RECORDED_SOURCES.md`).
3. Register an End User, approve it as Admin, sign in, upload a calibration bundle
   (`scripts/make_calibration_bundle.py`, see `docs/CALIBRATION_BUNDLE.md`) and start a session.
4. For LIVE input, point the camera rig (or `scripts/push_framesets.py`) at the ingest URL shown on the live view
   (`docs/LIVE_INGEST.md`).

For a quick look without MongoDB, `RETRACK_MONGO_URI=mongomock://localhost` uses an in-memory database (nothing is
saved, and the Admin must be created in the same process, so this is mainly for tests).

### Production

Run a single process behind an HTTPS reverse proxy with `RETRACK_ENV=production`, `RETRACK_REQUIRE_HTTPS=true` and
a long random `RETRACK_SECRET_KEY`. Details, retention policy and backup/recovery: `docs/OPERATIONS.md`.

## Tests

```bash
python -m pytest
```

These are web-system tests with an in-memory database and a fake model (NFR-13). They do not measure model accuracy
or performance; NFR-05/06/09 evidence must be recorded on the reference host (`docs/OPERATIONS.md`).
