# Report CSV schema (FR-26, FR-27)

UTF-8, comma-separated, one header row:

```
recordType,name,value,frameIndex,sourceTime,occupancy,trackIds
```

## `summary` rows (`name`, `value` filled)

| name | value |
|---|---|
| reportId, sessionId | identifiers |
| finalStatus | `Completed` or `Failed` |
| partialCoverage | `yes` for a Failed session (figures cover only frame sets committed before the failure) |
| inputMode | `LIVE` or `RECORDED` |
| generatedAt | generation time, ISO 8601 UTC |
| reportType | `AUTOMATIC` or `MANUAL` |
| firstFrameIndex, lastFrameIndex, firstSourceTime, lastSourceTime | source coverage of processed frame sets |
| framesRejected, framesDiscardedAtStop | input that was not processed |
| framesProcessed | number of committed processed frame sets |
| uniqueTracks | number of distinct **session-local track IDs** (not unique real people) |
| meanOccupancy | mean observed tracks per processed frame set, 3 decimals |
| peakOccupancy | largest observed tracks in one processed frame set |
| occupancyUnit | `observed tracks per processed frame set` |
| checkpointId, modelName, modelVersion, checkpointSha256, calibrationId, cameraOrder | model and calibration used |
| timeConvention | `ISO 8601, UTC` |

## `frame` rows (`frameIndex`, `sourceTime`, `occupancy`, `trackIds` filled)

One row per committed processed frame set, in increasing frame index. `trackIds` lists the observed track IDs
separated by `;` (empty when no pedestrians were observed, occupancy `0`).

Reproducing the summary from frame rows: `framesProcessed` = number of frame rows; `meanOccupancy` = mean of
`occupancy`; `peakOccupancy` = max of `occupancy`; `uniqueTracks` = number of distinct IDs across all `trackIds`.
Ground positions are not exported (NFR-08 data minimization); they remain in the database for authorized use.
