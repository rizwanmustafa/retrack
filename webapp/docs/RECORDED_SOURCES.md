# RECORDED sources (FR-14)

RECORDED sessions replay a folder on the server. The available recordings are listed in the JSON file named by
`RETRACK_RECORDED_SOURCES_FILE` (default `instance/recorded_sources.json`; an example is in
`instance/recorded_sources.example.json`):

```json
{
  "sources": [
    {
      "id": "wildtrack-test",
      "name": "WILDTRACK test split (frames 1800-1995)",
      "path": "/data/Wildtrack/Image_subsets",
      "cameraFolders": {"CVLab1": "C1", "CVLab2": "C2", "CVLab3": "C3", "CVLab4": "C4",
                        "IDIAP1": "C5", "IDIAP2": "C6", "IDIAP3": "C7"},
      "frameRate": 10,
      "startTime": "2017-01-01T00:00:00Z",
      "frameRange": [1800, 1995],
      "replayRate": 0
    }
  ]
}
```

| Field | Meaning |
|---|---|
| `path` | Folder with one sub-folder per camera; images are named by frame index (`00001800.png`) |
| `cameraFolders` | Sub-folder of each camera (default `C1`–`C7` in the configured camera order) |
| `frameRate` | Source frames per second; source time = `startTime + frameIndex / frameRate` |
| `startTime` | Source time of frame 0 (ISO 8601 with offset). WILDTRACK does not publish capture times, so this value is supplied by the operator; the example value is a placeholder. |
| `frameRange` | Optional inclusive range of frame indices to replay |
| `replayRate` | Frame sets per second to read; `0` reads as fast as inference allows |

Frame sets are read in increasing frame index. Incomplete or invalid frame sets are skipped and recorded; the
session finalizes as Completed at the end of the source.
