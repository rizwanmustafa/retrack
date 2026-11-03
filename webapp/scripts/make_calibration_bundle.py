"""
Builds a Retrack calibration bundle (.zip) from a WILDTRACK `calibrations` folder.

    python scripts/make_calibration_bundle.py --calibrations /data/Wildtrack/calibrations --out wildtrack-calib.zip

The WILDTRACK extrinsic translations are in centimetres on the WILDTRACK ground plane, which is what the manifest
declares. The bundle is validated by the web application when it is uploaded (UC-09).
"""
import argparse
import json
import os
import sys
import zipfile

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from retrack_web import scene  # noqa: E402
from retrack_web.calibration import extrinsic_name, intrinsic_name  # noqa: E402


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('--calibrations', required=True, help='WILDTRACK calibrations folder')
    p.add_argument('--out', required=True, help='output .zip')
    args = p.parse_args()

    manifest = {'coordinateConvention': scene.COORDINATE_CONVENTION, 'translationUnit': scene.TRANSLATION_UNIT}
    with zipfile.ZipFile(args.out, 'w', zipfile.ZIP_DEFLATED) as z:
        z.writestr('manifest.json', json.dumps(manifest, indent=1))
        for camera in scene.CAMERA_ORDER:
            for name in (intrinsic_name(camera), extrinsic_name(camera)):
                path = os.path.join(args.calibrations, name)
                if not os.path.isfile(path):
                    sys.exit(f'missing {path}')
                z.write(path, name)
    print(f'wrote {args.out}')


if __name__ == '__main__':
    main()
