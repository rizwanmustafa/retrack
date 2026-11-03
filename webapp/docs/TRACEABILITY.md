# Requirements traceability

Paths are relative to `webapp/`. Tests are in `tests/`.

## Functional requirements

| Req | Implementation | Tests |
|---|---|---|
| FR-01 Registration | `services/accounts.py: register_end_user`, `blueprints/auth.py: register`, `templates/auth/register*.html` | `test_accounts.py`: registration_creates_pending…, duplicate…, invalid_field…, cannot_create_or_promote_admin |
| FR-02 Account access | `security.py: authenticate, load_principal` | `test_accounts.py`: pending_and_rejected_accounts_cannot_sign_in |
| FR-03 Login | `blueprints/auth.py: login`, `security.py: start_auth_session, dashboard_endpoint` | `test_accounts.py`: invalid_credentials…, login_opens_only_the_role_dashboard |
| FR-04 Protected access | `security.py: login_required`; ownership checks in `blueprints/enduser.py`, `reports/service.py`, `monitoring/engine.py` | `test_accounts.py`: end_user_cannot_call_admin…, protected_requests…; `test_reports.py`: search_respects_access…; `test_monitoring.py`: live_result_structure_and_access |
| FR-05 Logout | `blueprints/auth.py: logout`, `security.py: end_auth_session` | `test_accounts.py`: logout_invalidates…; `test_monitoring.py`: logout_does_not_stop_the_running_session |
| FR-06 Account approval | `services/accounts.py: decide_account`, `blueprints/admin.py`, `templates/admin/accounts.html` | `test_accounts.py`: account_decision_is_made_once_and_recorded |
| FR-07 Profile request | `services/profiles.py: submit_request`, `templates/enduser/profile_requests.html` | `test_profiles.py`: request_needs_fields_and_reason |
| FR-08 Profile decision | `services/profiles.py: decide_request`, `templates/admin/profile_request*.html` | `test_profiles.py`: admin_decision_records…, rejected_request_is_visible… |
| FR-09 Approved editing | `services/profiles.py: apply_update`, `templates/enduser/profile.html` | `test_profiles.py`: approved_fields_update_once…, unauthorized_field_change…, role_and_status…, invalid_or_duplicate… |
| FR-10 Calibration input | `services/calibrations.py`, `templates/enduser/setup.html` | `test_calibration_checkpoints.py`: valid_bundle…, session_retains_calibration…, stored_calibration_can_be_selected… |
| FR-11 Calibration validation | `calibration.py: validate_bundle`, `services/calibrations.py: revalidate` | `test_calibration_checkpoints.py`: missing_file…, malformed_matrix…, convention_and_unit… |
| FR-12 Checkpoint administration | `services/checkpoints.py`, `ai/tracktacular.py: inspect_checkpoint`, `templates/admin/checkpoints.html` | `test_calibration_checkpoints.py`: incompatible…, unavailable…, change_refused_while_active…; `test_tracktacular_adapter.py` |
| FR-13 Session initialization | `monitoring/engine.py: start, _run, _fail_initialization` | `test_monitoring.py`: initialization_failure…, each_session_resets_tracking |
| FR-14 Input modes | `monitoring/feeds.py: RecordedFeed, LiveFeed` | `test_monitoring.py`: recorded_replay…, live_ingest… |
| FR-15 Frame validation | `monitoring/frameset.py: build_frame_set` | `test_monitoring.py`: live_ingest…, mismatched_per_camera_metadata… |
| FR-16 Ordered processing | `monitoring/frameset.py: check_order`, `feeds.py` | `test_monitoring.py`: recorded_replay… (order), live_ingest… (duplicate / out of order) |
| FR-17 Input faults | `monitoring/engine.py: record_input_fault` | `test_monitoring.py`: live_ingest… (pause and resume), recorded_replay… |
| FR-18 Pedestrian inference | `ai/tracktacular.py: analyze` | needs the inference environment (manual acceptance) |
| FR-19 Temporal tracking | `ai/tracktacular.py: reset_tracking, analyze`; `monitoring/engine.py: _commit` (Track records) | `test_monitoring.py`: zero_detection_frame… (trajectory, status) |
| FR-20 Result persistence | `monitoring/engine.py: _commit` | `test_monitoring.py`: zero_detection_frame_is_committed_with_occupancy_zero |
| FR-21 Tracking display | `templates/enduser/live.html`, `static/js/live.js`, `calibration.py: project_ground_point` | `test_monitoring.py`: live_result_structure_and_access |
| FR-22 Structured result | `monitoring/engine.py: snapshot`, `blueprints/api.py: live_result` | `test_monitoring.py`: live_result_structure_and_access |
| FR-23 Session completion | `monitoring/engine.py: stop, _finalize` | `test_monitoring.py`: stop_while_waiting…, recorded_replay… |
| FR-24 Processing failure | `monitoring/engine.py: _loop, _close_session` | `test_monitoring.py`: fatal_failure…, interrupted_sessions… |
| FR-25 Report generation | `reports/service.py: generate, generate_manual` | `test_reports.py`: manual_generation_rules, render_failure… |
| FR-26 Report summary | `reports/summary.py` | `test_monitoring.py`: zero_detection…; `test_reports.py`: pdf_and_csv… |
| FR-27 Report formats | `reports/render.py`, `docs/REPORT_CSV_SCHEMA.md` | `test_reports.py`: pdf_and_csv_carry_the_same_summary, partial_report…, render_failure… |
| FR-28 Report retrieval | `reports/service.py: search`, `blueprints/reports.py`, `templates/reports/*.html` | `test_reports.py`: search_respects_access_and_date_boundaries |
| FR-29 Report download | `reports/service.py: open_file`, `blueprints/reports.py: download` | `test_reports.py`: downloads_are_checked_and_audited |
| FR-30 Audit and diagnostics | `audit.py`, `blueprints/admin.py: audit_log, audit_detail` | `test_reports.py`: audit_log_covers_activity_and_contains_no_credentials |

## Non-functional requirements

| Req | How it is met |
|---|---|
| NFR-01 Usability | Status badges and distinct alerts for waiting / paused / processing / failed / finalized (`live.js`); problems list camera, file, field and reason; zero pedestrians shown as a valid state; viewing never blocks processing |
| NFR-02 Credentials | Salted scrypt hashes; hashed session tokens; `audit._clean` strips secret keys (`test_security.py`, `test_reports.py`) |
| NFR-03 Permissions | All checks are server-side (`security.login_required`, ownership checks, CSRF) (`test_security.py`) |
| NFR-04 Transport | `RETRACK_REQUIRE_HTTPS`, Secure/HttpOnly cookies, HSTS, generic error pages, `textutil.safe_message` (`test_security.py`) |
| NFR-05, NFR-06 Performance | Stop/Start acknowledged immediately (work continues in the worker); per-frame `receivedAt`, `committedAt`, `processingMs` stored; `scripts/measure_session.py` — **must be measured on the reference host** |
| NFR-07 Capacity | `MonitoringEngine.is_busy`, one process (`docs/OPERATIONS.md`) (`test_monitoring.py`: second_start…) |
| NFR-08 Privacy | No images stored; no identity fields; positions not exported in CSV; retention policy in `docs/OPERATIONS.md` |
| NFR-09 Compatibility | Bootstrap 5.3 (Chromium and Firefox); tested versions to be recorded during acceptance |
| NFR-10 Persistence | MongoDB + GridFS; a report becomes Available only after both files are stored |
| NFR-11 Reliability | Tracking reset per session; failures keep committed results; zero detections are valid results |
| NFR-12 Reproducibility | Session stores repository revision, environment, checkpoint SHA-256, calibration, camera order; thresholds from the checkpoint |
| NFR-13 Evaluation honesty | Tests are labelled web-system tests; no accuracy or real-time claim in the UI |
| NFR-14 Maintainability | Web (`blueprints`, `templates`), persistence (`db.py`, `services`), model integration (`ai/`) separated; `AIModel.loadCheckpoint / resetTracking / analyze` |
| NFR-15 Time and units | `timeutil.py` (UTC ISO 8601), cm on the WILDTRACK ground plane, declared in every result |
| NFR-16 Recovery | `docs/OPERATIONS.md` backup/restore; `MonitoringEngine.recover_interrupted` |

## Use cases

| UC | Pages / endpoints |
|---|---|
| UC-01 Register Account | `/register` |
| UC-02 Log In / Log Out | `/login`, `POST /logout` |
| UC-03 Authenticate User | every request (`security.load_principal`, `login_required`) |
| UC-04 Request Profile Update | `/user/profile/requests` |
| UC-05 Update Profile | `/user/profile` |
| UC-06 Approve / Reject Accounts | `/admin/accounts` |
| UC-07 Approve / Reject Profile Requests | `/admin/profile-requests`, `/admin/profile-requests/<id>` |
| UC-08 Upload / Select Calibration | `/user/setup` (Calibration step) |
| UC-09 Validate Calibration | on upload and selection |
| UC-10 Manage Model Checkpoint | `/admin/checkpoints` |
| UC-11 Start / Stop Monitoring Session | `/user/setup` → Start; Stop on the live view |
| UC-12 Supply Synchronized Frames | `POST /api/ingest/sessions/<id>/framesets` (LIVE); recorded replay |
| UC-13 View Live Tracking Results | `/user/sessions/<id>/live`, `GET /api/sessions/<id>/live` |
| UC-14 Generate Session Report | automatic at finalization; "Generate report" on the session page |
| UC-15 Search Reports | `/reports` |
| UC-16 Download Report | `/reports/<id>/download/pdf`, `/reports/<id>/download/csv` |
| UC-17 View Audit Log and System Errors | `/admin/audit`, `/admin/audit/<id>` |
