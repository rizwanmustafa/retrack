# SRS assumptions

Wherever the Retrack SRS leaves room for interpretation, the web application had to make a choice. Every such choice
is listed here with the SRS text that left it open, so it can be confirmed or changed by the team. Contradictions
*within* the SRS are listed separately in [SRS_CONTRADICTIONS.md](SRS_CONTRADICTIONS.md).

Each entry: **what was decided** — *why / where the SRS is silent*.

## A. Accounts and authentication (FR-01 to FR-06, UC-01 to UC-03, UC-06)

| # | Decision | Reason |
|---|---|---|
| A-1 | Sign-in uses **email + password**. | Figure 12 shows `Login(email, password)`; FR-03/UC-02 only say "credentials". |
| A-2 | Field rules: name 1–100 characters; username 3–32 of `A–Z a–z 0–9 . _ -`; email of the form `x@y.z`, max 254; password 8–128 characters. | FR-01 requires validation but defines no rules. |
| A-3 | Usernames and emails are unique **ignoring case**, and across End User *and* Admin accounts. | FR-01 says "reject duplicate username or email"; case and cross-role uniqueness are not specified. Cross-role uniqueness keeps email sign-in unambiguous. |
| A-4 | The browser authentication session expires after **30 minutes without activity** (configurable, `RETRACK_AUTH_IDLE_TIMEOUT_MINUTES`). Sessions are stored server-side (hashed token), so logout really invalidates them. | UC-02 3a mentions expiry without a duration. FR-05 requires invalidation. |
| A-5 | Admin accounts are provisioned with the command `flask --app wsgi create-admin`. An Admin has username, email and password (no name). | "Admin accounts are provisioned separately" (UC-01); no mechanism given. Fields follow the class diagram. |
| A-6 | There is **no password change or reset** feature, and password is not a requestable profile field. | No requirement or use case covers it. |
| A-7 | A Pending/Rejected account message is shown only after the password has been verified; a wrong password always gives the generic "Invalid email or password". | UC-03 checks credentials (step 1) before status (step 2); FR-02 forbids exposing password information. |
| A-8 | Pending registrations stay Pending until an Admin decides; there is no expiry. | Not specified. |

## B. Profile requests (FR-07 to FR-09, UC-04, UC-05, UC-07)

| # | Decision | Reason |
|---|---|---|
| B-1 | Requestable fields: **name, username, email**. | UC-04 says "the profile fields that can be requested" without listing them; role and status are excluded by the SRS. |
| B-2 | A user can have **one open request** (Pending, or Approved and unused) at a time. | Not specified; otherwise "an unused Approved request" in FR-09 would be ambiguous when several exist. |
| B-3 | The Admin may approve a **subset** of the requested fields (at least one). | FR-08 records "permitted fields" separately from the requested fields. |
| B-4 | A rejection carries no comment from the Admin. | Not specified. |
| B-5 | Applying an approved update is not written to the audit log. | FR-30 lists "account and profile decisions", not profile edits. |

## C. Calibration (FR-10, FR-11, UC-08, UC-09)

| # | Decision | Reason |
|---|---|---|
| C-1 | Bundle format: a `.zip` with `manifest.json` plus the WILDTRACK layout `intrinsic_zero/intr_<camera>.xml` (OpenCV `camera_matrix` 3×3) and `extrinsic/extr_<camera>.xml` (`rvec` Rodrigues rotation, `tvec` translation). See `docs/CALIBRATION_BUNDLE.md`. | The SRS says "calibration bundle" without a format. This is the format the backend (WorldTrack) already reads. |
| C-2 | The convention and unit are **declared** in the manifest (`coordinateConvention: "WILDTRACK_GROUND_PLANE"`, `translationUnit: "cm"`); optional per-camera declarations must match. | FR-11 requires "a consistent coordinate convention and translation unit", but WILDTRACK files do not record them. |
| C-3 | Invalid bundles are not stored; the problems are shown (camera, file, field, reason). | UC-09: "rejected with a list of problems". |
| C-4 | Every valid stored calibration is selectable by **every** End User (the uploader is shown). | UC-08 says "select a stored one" without ownership; the rig is shared by all operators. |
| C-5 | The selected calibration is remembered per user until changed, and is re-validated when selected and at Start. | Not specified. |
| C-6 | Validation checks structure, dimensions, finiteness, convention and unit only: no distortion coefficients, no plausibility of focal lengths. | UC-09 lists exactly these checks and says validation cannot prove a match with the scene. |

## D. Model checkpoints (FR-12, UC-10)

| # | Decision | Reason |
|---|---|---|
| D-1 | Checkpoints are `.ckpt` files placed in `RETRACK_CHECKPOINT_DIR` by the team; the page lists them by scanning that folder. There is no browser upload. | "Trained checkpoints are available" (UC-10 precondition); checkpoint preparation is an offline activity (2.4). |
| D-2 | Compatibility = 7 cameras **and** BEV resolution/bounds equal to `configs/d_wildtrack.yml` (scene) **and** a supported TrackTacular model (segnet, mvdet, liftnet, bevformer) **and** the weights load strictly into the backend (output contract). | FR-12 names the four aspects without defining them. |
| D-3 | Checkpoint "version" = Lightning epoch and step; checkpoint identity = SHA-256 of the file. | Not defined; NFR-12 needs a checkpoint identity. |
| D-4 | Detection threshold, tracker type and re-ID settings come from the checkpoint's saved hyper-parameters. | NFR-12: web replay and offline comparison use the same thresholds. |
| D-5 | One checkpoint is active system-wide; a change applies to sessions started afterwards. | UC-10: "active for new sessions". |

## E. Sessions and input (FR-13 to FR-24, UC-11 to UC-13)

| # | Decision | Reason |
|---|---|---|
| E-1 | **One model instance per server process.** The server must run as a single process (threads are fine). | C-06/NFR-07: one active session per model instance; the deployment layout is not specified. |
| E-2 | LIVE transport: the camera rig sends each frame set as one HTTP `multipart/form-data` POST with `frameIndex`, `sourceTime` and one image per camera (field name = camera ID, in the configured order), authenticated by a per-session **ingest token** shown to the session owner. Responses follow Figure 8: `frameSetAccepted(frameIndex)` / `frameSetRejected(frameIndex, reason)`. See `docs/LIVE_INGEST.md`. | The SRS defines the contract (Figure 8) but not the transport ("does not assert support for every commercial camera transport protocol"). The Camera Rig has no account, so it needs its own credential. |
| E-3 | `sourceTime` must carry a UTC offset or `Z`. Both frame index and source time must strictly increase; a repeated index is `DUPLICATE_FRAME`, a lower index or non-increasing time is `OUT_OF_ORDER`. Optional per-camera `frameIndex.<camera>` / `sourceTime.<camera>` fields must match the set (otherwise `MISMATCHED_METADATA`). | FR-15/16 require a common index/time and increasing order without defining duplicates or time zones. |
| E-4 | Accepted LIVE frame sets wait in a queue of 8 (configurable). When it is full, the frame set is rejected as `BACKLOG_FULL` and recorded. | Behaviour under overload is not specified; this keeps processing in source order without unbounded memory. |
| E-5 | At Stop, accepted but not yet processed frame sets are discarded and counted (`framesDiscarded`, shown in reports). | FR-23: "stop accepting new input"; queued input is not addressed. |
| E-6 | RECORDED sources are configured per deployment in a JSON file (id, name, folder, camera folders `C1`–`C7`, `frameRate`, `startTime`, optional `frameRange`, `replayRate`). Source time = `startTime + frameIndex / frameRate`. Replay is not paced unless `replayRate` is set. See `docs/RECORDED_SOURCES.md`. | The SRS does not say where recordings come from or how their source time is derived. WILDTRACK does not publish absolute capture times, so `startTime` must be provided by the operator. |
| E-7 | A recorded frame with a missing camera image is skipped and recorded, and the session shows "Paused: missing camera" until the next complete valid set. | FR-17 (pause) applied to replay, where the next set is read immediately. |
| E-8 | A recorded source that contains no frames makes initialization fail (session Failed, `INITIALIZATION_FAILED`). | Not specified. |
| E-9 | Every rejected frame set creates an `INPUT_ERROR` audit record. | FR-17 "skipped and recorded", FR-30 "input errors". |
| E-10 | Session statuses: `Initializing`, `Running`, `Completed`, `Failed`; a Running session also shows `Waiting for input`, `Processing`, `Paused: missing camera` or `Stopping`. A failed initialization ends as `Failed` with end reason `INITIALIZATION_FAILED`. | The SRS names Running, Completed, Failed and the waiting/fault states (NFR-01) but no status for initialization. |
| E-11 | Stop is acknowledged at once; finalization (and the automatic report) continues in the background, and the live view shows the result. | NFR-05 (≤2 s acknowledgement, excluding report rendering). |
| E-12 | If the server stops during a session, that session is marked `Failed` (`SERVER_RESTART`) at the next start-up, committed results are kept and a partial report is attempted. | NFR-16: automatic resumption is out of scope; the state of interrupted sessions is not specified. |
| E-13 | **Occupancy = number of tracks the tracker reports as observed in that frame set** (matched or newly confirmed). Lost/predicted tracks and unconfirmed detections do not count. | FR-26: "unobserved predicted tracks shall not increase occupancy"; the exact counting rule is not given. |
| E-14 | Track status: `Tracked` when observed, `Lost` when not observed in the next frame set, `Ended` at finalization. | Class diagram has `TrackStatus` without values. |
| E-15 | The live view polls every second. Camera images are shown as 480×270 JPEG thumbnails, kept in memory only while someone is viewing. Annotations are a marker and track ID at the projected ground position (no boxes). | FR-21 does not define the update mechanism or annotation style; NFR-08 limits image retention; see X-1. |
| E-16 | Admins can open session records read-only (to generate reports and read diagnostics) but cannot start, stop or watch sessions. | Role table: monitoring "Not assigned" to Admin; FR-25 lets Admin generate reports for sessions. |
| E-17 | Identifiers are readable sequences: `U-000001` (user), `A-` (admin), `P-` (profile request), `C-` (calibration), `K-` (checkpoint), `S-` (session), `R-` (report), `L-` (audit record), `F-` (feed). | No format given; session identifiers are typed into the report search. |
| E-18 | The backend's temporal-cache frame number is the count of processed frame sets in the session. | The research code uses dataset frame numbers, which do not exist for LIVE input. |

## F. Reports (FR-25 to FR-29, UC-14 to UC-16)

| # | Decision | Reason |
|---|---|---|
| F-1 | Every generation creates a new report (`AUTOMATIC` or `MANUAL`); a session can have several. | FR-25 allows manual generation and retries without saying whether earlier reports are replaced. |
| F-2 | Search: session identifier is an exact match; the date range filters on **report generation time**, as whole UTC days, both ends inclusive. | FR-28 does not say which date is filtered or how matching works. |
| F-3 | An Admin's "authorized reports" are **all** reports. | Role table: "Authorized reports" without a definition. |
| F-4 | Report files are stored in MongoDB GridFS. | NFR-10 requires persistent storage outside the inference runtime; MongoDB is the chosen store. |
| F-5 | CSV is one table with a `recordType` column (`summary` rows, then one `frame` row per processed frame set with index, source time, occupancy and observed track IDs). Positions are not exported. See `docs/REPORT_CSV_SCHEMA.md`. | FR-27 requires summary and frame-level data sufficient to reproduce occupancy; NFR-08 favours minimal movement data. |
| F-6 | `meanOccupancy` is shown with 3 decimal places. | Precision not specified. |
| F-7 | "Source coverage" = first and last processed frame index and source time, plus the counts of rejected and discarded frame sets, plus a partial flag for Failed sessions. | FR-26 names "source coverage" without defining it. |
| F-8 | The PDF contains the summary, an occupancy-per-frame chart and the identification table; frame-level data is in the CSV only. | Contents beyond FR-26 are not specified. |

## G. Audit (FR-30, UC-17)

| # | Decision | Reason |
|---|---|---|
| G-1 | Recorded activity types: account decisions (and refused second decisions), profile decisions (and refusals), checkpoint changes (and refusals), session started / start failed or refused / completed / failed, report generated / generation failed / downloaded, input errors, processing errors. Sign-in and sign-out are not recorded. | FR-30's list; sign-in/out is not in it. |
| G-2 | The audit page filters by activity type or group, resource ID, date range (UTC days) and "errors only", 50 records per page. | UC-17 step 2: "filters the records by type or time". |
| G-3 | Records are never deleted by the application. | No retention rule is given for audit records. |

## H. Quality and deployment (NFR-01 to NFR-16)

| # | Decision | Reason |
|---|---|---|
| H-1 | Times are stored and shown in UTC as `YYYY-MM-DDTHH:MM:SS.mmmZ`. | NFR-15 requires "a documented ISO 8601 convention". |
| H-2 | Ground positions are WILDTRACK world coordinates in cm; the BEV is drawn with world *y* (36 m) horizontal and world *x* (12 m) vertical. | NFR-15 requires a declared system; orientation of the display is not specified. |
| H-3 | HTTPS is terminated by a reverse proxy; with `RETRACK_REQUIRE_HTTPS=true` the app redirects HTTP to HTTPS, sets Secure cookies and HSTS, and refuses non-GET HTTP requests. | NFR-04 requires HTTPS outside local development; the setup is not specified. |
| H-4 | No automatic deletion is implemented. The retention and deletion policy is a documented deployment decision (`docs/OPERATIONS.md`), with a proposed default. | NFR-08 requires the policy to be documented, not a particular period. |
| H-5 | Backup and recovery use `mongodump` / `mongorestore` (this covers GridFS report files). | NFR-16 requires a documented procedure. |
| H-6 | Bootstrap 5.3.3 and Bootstrap Icons 1.11.3 load from the jsDelivr CDN with integrity hashes, so the browser needs internet access. | The stack is fixed as Bootstrap; hosting is not specified. |
| H-7 | The light/dark theme switch from the Bootstrap examples is included. | Not required; it comes with the Bootstrap templates used. |
| H-8 | Automated tests use an in-memory MongoDB (mongomock) and a fake model. They are **web-system tests**; the NFR-05/06/09 measurements (latency, throughput, browser versions) must still be taken on the reference host. | NFR-13 separates offline model evaluation from web-system tests; this machine has no GPU, torch or checkpoints. |
| H-9 | CSRF tokens protect every form. | Not in the SRS; part of the NFR-03 intent that requests cannot be forged to bypass checks. |
