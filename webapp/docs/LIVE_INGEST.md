# LIVE input contract (FR-14 to FR-17, UC-12, Figure 8)

When an End User starts a LIVE session, the live view shows the ingest URL and an **ingest token** for that session.
The camera rig supplies one frame set per request:

```
POST /api/ingest/sessions/<sessionId>/framesets
Authorization: Bearer <ingest token>
Content-Type: multipart/form-data

frameIndex   = 1805                          (whole number, strictly increasing)
sourceTime   = 2026-11-03T10:03:00.500Z      (ISO 8601 with Z or an offset, strictly increasing)
CVLab1       = <image file>                  one 1920x1080 image per camera, in this order:
CVLab2, CVLab3, CVLab4, IDIAP1, IDIAP2, IDIAP3
frameIndex.<camera>, sourceTime.<camera>     optional; if sent they must equal the set's values
```

## Responses

| Status | Body | Meaning |
|---|---|---|
| 202 | `{"frameSetAccepted": {"frameIndex": 1805}}` | Validated and queued for inference |
| 422 | `{"frameSetRejected": {"frameIndex": 1805, "kind": "MISSING_CAMERA", "reason": "…", "cameras": ["IDIAP2"]}}` | Skipped and recorded; not counted in framesProcessed |
| 401 | `{"error": "invalid_token"}` | Wrong token |
| 409 | `{"error": "session_not_running"}` | The session is initializing, stopped or finished |

Rejection kinds: `MISSING_CAMERA` (inference pauses until a complete set arrives), `DUPLICATE_CAMERA`,
`UNKNOWN_CAMERA`, `CAMERA_ORDER`, `UNDECODABLE`, `WRONG_SIZE`, `MISMATCHED_METADATA`, `BAD_METADATA`,
`DUPLICATE_FRAME`, `OUT_OF_ORDER`, `BACKLOG_FULL`.

Arrival time is never used to pair images: a frame set is exactly the images sent in one request with one declared
frame index and source time.

## Test client

`scripts/push_framesets.py` acts as a camera rig and replays a WILDTRACK `Image_subsets` folder:

```
python scripts/push_framesets.py --url http://localhost:5000 --session S-000004 --token <token> \
    --images /data/Wildtrack/Image_subsets --start-time 2026-11-03T10:00:00Z --frame-rate 10 --rate 2
```
