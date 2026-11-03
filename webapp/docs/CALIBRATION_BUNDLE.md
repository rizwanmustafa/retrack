# Calibration bundle format (FR-10, FR-11)

A calibration bundle is a `.zip` file:

```
manifest.json
intrinsic_zero/intr_CVLab1.xml   … intr_IDIAP3.xml     (7 files)
extrinsic/extr_CVLab1.xml        … extr_IDIAP3.xml     (7 files)
```

The files may also sit inside one top-level folder (for example `calibrations/`).

## manifest.json

```json
{"coordinateConvention": "WILDTRACK_GROUND_PLANE", "translationUnit": "cm"}
```

Optional per-camera declarations are allowed, and must agree with the bundle-level values:

```json
{"coordinateConvention": "WILDTRACK_GROUND_PLANE", "translationUnit": "cm",
 "cameras": {"IDIAP3": {"translationUnit": "cm"}}}
```

## Intrinsic file (OpenCV storage)

`camera_matrix` must be 3×3 with nine finite numbers:

```xml
<opencv_storage>
<camera_matrix type_id="opencv-matrix">
  <rows>3</rows><cols>3</cols><dt>d</dt>
  <data>fx 0 cx 0 fy cy 0 0 1</data></camera_matrix>
</opencv_storage>
```

## Extrinsic file

`rvec` is a Rodrigues rotation vector and `tvec` a translation in cm, three finite numbers each:

```xml
<opencv_storage>
<rvec>1.759 0.467 -0.332</rvec>
<tvec>-525.9 45.4 986.7</tvec>
</opencv_storage>
```

## Building a bundle from WILDTRACK

```
python scripts/make_calibration_bundle.py --calibrations /data/Wildtrack/calibrations --out wildtrack-calib.zip
```

## What validation checks

One intrinsic and one extrinsic file per camera, the matrix and vector sizes, finite values, the declared
convention and unit. Every problem is reported with the camera, file, field and reason, and Start stays unavailable.
Validation cannot prove that the calibration matches the physical scene; moving a camera invalidates its calibration.
