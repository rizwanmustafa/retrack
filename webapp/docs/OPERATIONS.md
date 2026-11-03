# Operations: deployment, privacy, backup and acceptance evidence

## Deployment (NFR-04, NFR-07, NFR-10, C-07)

* Run **one** application process (threads are fine). The process owns the single model instance; several worker
  processes would each load a model and break the one-active-session rule.
  Example: `waitress-serve --threads=8 --call wsgi:create_app`.
* MongoDB must be persistent storage outside the GPU runtime (for example a MongoDB server or Atlas, not a
  temporary Colab disk). Session records, frame records and report files (GridFS) all live there.
* Terminate HTTPS at a reverse proxy (nginx, Caddy) and set `RETRACK_REQUIRE_HTTPS=true` and a long random
  `RETRACK_SECRET_KEY`. The proxy must send `X-Forwarded-Proto: https`.
* Client-facing errors never contain stack traces or server paths; details are in the server log.

## Privacy, retention and deletion (NFR-08)

* Only End Users (their own sessions) and Admins can see tracking results and reports.
* There is no face recognition and no real-identity field. Track IDs are session-local.
* Raw images are never written to storage. The latest frame set's thumbnails exist only in memory while viewed.
* Stored ground positions and track IDs are **sensitive movement data**, not anonymous data.
* **Retention policy (proposed default, to be confirmed by the deployment owner):** frame records, tracks and reports
  are kept for 90 days after a session ends, audit records for 1 year. The application does not delete data
  automatically; deletion is an operator task, for example:

  ```js
  // mongosh: delete sessions that ended before a cut-off date, with their results and reports
  const cutoff = ISODate("2026-08-01T00:00:00Z");
  const ids = db.sessions.find({endTime: {$lt: cutoff}}).map(s => s._id);
  db.frame_results.deleteMany({sessionId: {$in: ids}});
  db.tracks.deleteMany({sessionId: {$in: ids}});
  const reports = db.reports.find({sessionId: {$in: ids}}).map(r => r._id);
  db.report_files.files.find({"metadata.reportId": {$in: reports}}).forEach(f => {
    db.report_files.chunks.deleteMany({files_id: f._id}); db.report_files.files.deleteOne({_id: f._id}); });
  db.reports.deleteMany({_id: {$in: reports}});
  db.sessions.deleteMany({_id: {$in: ids}});
  ```

## Backup and recovery (NFR-16)

* Backup (daily, plus before upgrades): `mongodump --uri "$RETRACK_MONGO_URI" --db retrack --gzip --archive=retrack-$(date +%F).gz`
  This includes accounts, sessions, frame records, audit records and report files (GridFS).
* Restore: stop the application, then `mongorestore --uri "$RETRACK_MONGO_URI" --gzip --archive=retrack-YYYY-MM-DD.gz --drop`,
  then start it.
* Interrupted inference is not resumed. On start-up the application marks sessions that were active as Failed
  (`SERVER_RESTART`), keeps their committed results and attempts a partial report.
* Checkpoint files are not in the database; back up `RETRACK_CHECKPOINT_DIR` separately.

## Acceptance evidence (NFR-05, NFR-06, NFR-09, NFR-12, NFR-13)

Web-system tests (`pytest`) use an in-memory database and a fake model; they show workflow behaviour, not model
accuracy. Offline model evaluation is in `WorldTrack/evaluation/` and must be reported separately.

Record for each acceptance run:

| Item | Where it comes from |
|---|---|
| Repository revision, inference environment, device | Session page → "Inference environment" (stored with the session) |
| Checkpoint identity (SHA-256) and configuration | Admin → Model checkpoints; report summary |
| Throughput and p95 result delay after warm-up (NFR-06) | `python scripts/measure_session.py --session S-…` |
| Command acknowledgement p95 (NFR-05) | Time `POST /api/sessions/<id>/stop` and the Start form submission (e.g. browser DevTools or `curl -w %{time_total}`) |
| Browser versions (NFR-09) | Chromium-based and Firefox versions used for the account, monitoring and reporting workflows |
| Source rate and load | LIVE: `--rate` of the camera rig client; RECORDED: `replayRate` |
