# SRS contradictions

This file lists places where parts of the Retrack SRS (`Retrack_SRS.docx.pdf`) contradict each other, and the
resolution the web application implements. The rule applied throughout: **the functional requirements (FR-xx),
non-functional requirements (NFR-xx), constraints (C-xx) and expanded use cases (UC-xx) are authoritative**; the
diagrams (sections 9–15) are treated as illustrations and are followed wherever they do not conflict with them.

Gaps where the SRS is silent (rather than self-contradictory) are listed in [SRS_ASSUMPTIONS.md](SRS_ASSUMPTIONS.md).

---

## X-1 Tracking result structure: bounding boxes and global person IDs

| | |
|---|---|
| **Source A** | Figure 15 (JSON Tree): `TrackingResult` = `resultID, sessionID, cameraID, globalPersonID, boundingBox, timestamp` |
| **Source B** | FR-18, FR-19, FR-20, FR-22, C-09, UC-12, scope (2.4) |
| **Conflict** | Figure 15 describes per-camera bounding boxes with a *global person ID*. The requirements describe *ground positions* in a common bird's eye view, *session-local track IDs* that "shall not claim a real identity or continuity across separate sessions", confidence and frame occupancy. The scope excludes real-person identity and cross-session re-identification, and the backend (TrackTacular) outputs BEV ground points, not boxes. |
| **Resolution** | Followed the FRs. A committed frame record holds `sessionId, frameIndex, timestamp, observations[{trackId, x, y, confidence}], occupancy` (FR-20), and the structured response holds `session, model, frameSet, results` (FR-22). There is no `globalPersonID` and no `boundingBox`. Camera annotations (FR-21) are drawn by projecting each track's ground position into each camera with the session calibration. |

## X-2 Stored password vs. salted hash

| | |
|---|---|
| **Source A** | Figure 15: `Admin.password`, `User.password` |
| **Source B** | NFR-02, UC-01 postconditions, class diagram (`passwordHash`) |
| **Conflict** | The JSON tree suggests storing a password; NFR-02 requires salted one-way hashes. |
| **Resolution** | Only `passwordHash` (salted scrypt, Werkzeug) is stored. No password is stored, logged or returned. |

## X-3 Report search filters

| | |
|---|---|
| **Source A** | Figure 6 (SSD Search Report): `enterSearchFilters(date, time, location, type)` |
| **Source B** | FR-28, UC-15: search "using session identifiers and date ranges" |
| **Conflict** | The SSD adds *location* and *type* filters and omits the session identifier. The system has one fixed scene, so a location filter has nothing to select, and "type" is not defined. |
| **Resolution** | Followed FR-28/UC-15: session identifier and date range. Time-of-day filtering is not offered; date boundaries are whole UTC days (NFR-15). |

## X-4 Admin manual report generation

| | |
|---|---|
| **Source A** | Class diagram: `User.generateReport(feed)`; `Admin` has no generate operation |
| **Source B** | FR-25, UC-14 (actors End User, Admin): manual generation "from the owning End User or Admin" |
| **Resolution** | Followed FR-25: both the owning End User and an Admin can generate a report manually. |

## X-5 One report file vs. two formats

| | |
|---|---|
| **Source A** | Figure 15: `Report.filePath` (single) |
| **Source B** | Class diagram `filePaths: Map<Format, String>`, FR-27 (PDF and CSV) |
| **Resolution** | Each report has two files, PDF and CSV, with identical summary figures (FR-27). |

## X-6 "Real-time" claim

| | |
|---|---|
| **Source A** | Abstract: "a real-time multi-camera person detection and tracking system" |
| **Source B** | NFR-06: "If unmet, measured throughput and the supported replay rate shall be reported; real-time capability shall not be claimed"; NFR-13 |
| **Resolution** | The interface never claims real-time operation. "LIVE" is only the name of the input mode. `scripts/measure_session.py` produces the measured throughput and delay that NFR-06 asks for. |

## X-7 Camera images in the persistent domain model

| | |
|---|---|
| **Source A** | Domain model `FrameSet.cameraImages`, class diagram `FrameSet.images: Map<CameraId, Image>` |
| **Source B** | NFR-08: "Raw images shall not be retained automatically solely to generate session reports" |
| **Resolution** | FrameSets (with images) exist only in memory while they are validated and processed. The live view keeps small JPEG thumbnails of the latest frame set in memory, and only while someone is viewing. Only results are persisted. |

## X-8 Calibration owned by the rig vs. per session

| | |
|---|---|
| **Source A** | Figure 15: `CameraRig.calibrationID` (one calibration per rig) |
| **Source B** | FR-10, UC-08: each End User uploads or selects a calibration before starting, and "the session shall retain the calibration reference and camera order used" |
| **Resolution** | Each session stores its own `calibrationId` and `cameraOrder` (FR-10). The `camera_rigs` document's `calibrationID` records the calibration of the most recently started session, as a convenience, and nothing depends on it. |

## X-9 "Request Profile Update «include» Update Profile"

| | |
|---|---|
| **Source A** | Use case diagram and UC-04 type: "Primary (includes UC-05 Update Profile)" |
| **Source B** | UC-05 precondition (an Approved, unused request), UC-07 (Admin decision in between), FR-09 |
| **Conflict** | «include» means the included use case always runs as part of the base one. Here, Update Profile can only happen later, after a separate Admin decision, and only if the request was approved. |
| **Resolution** | Implemented as two separate steps: request (UC-04) → Admin decision (UC-07) → optional update (UC-05). |

## Incomplete data structures (Figure 15) — extended, not contradicted

Figure 15 omits fields that the FRs require. These were added (they do not contradict anything):

* `Session`: input mode, calibration reference, camera order, checkpoint identity, counters, end reason, failure
  diagnostic, summary (FR-10, FR-13, FR-22 to FR-24).
* `ProfileRequest`: permitted fields, decision time, used time (FR-08, FR-09).
* `AuditLog`: affected resource and outcome (FR-30).
* `Report`: owner, final session status, input mode, source coverage, summary figures, status (FR-26, FR-27).

## Editorial inconsistencies (no effect on behaviour)

* Table numbering differs between the "List of Tables" and the body (e.g. "Table 9 Role Permissions" vs.
  "Table 12 System Features").
* Figure 2 places "Record failure and notify operator / Admin" in the Camera / Input Source swimlane.
