"""
Calibration bundles (FR-10, FR-11, UC-08, UC-09).

A bundle is a .zip with the WILDTRACK calibration layout plus a manifest that declares the coordinate convention and
translation unit (docs/CALIBRATION_BUNDLE.md):

    manifest.json                      {"coordinateConvention": "WILDTRACK_GROUND_PLANE", "translationUnit": "cm"}
    intrinsic_zero/intr_<camera>.xml   OpenCV storage, camera_matrix 3x3
    extrinsic/extr_<camera>.xml        rvec (3 values, Rodrigues rotation) and tvec (3 values)

All files may also sit under one top-level folder. Validation checks structure and values only; it cannot prove that
the calibration matches the physical scene (UC-09 constraints).
"""
import hashlib
import io
import json
import math
import xml.etree.ElementTree as ET
import zipfile

import numpy as np

from . import scene

MAX_MEMBER_BYTES = 1024 * 1024


def intrinsic_name(camera):
    return f'intrinsic_zero/intr_{camera}.xml'


def extrinsic_name(camera):
    return f'extrinsic/extr_{camera}.xml'


def rodrigues(rvec):
    """Rotation vector -> 3x3 rotation matrix (same convention as OpenCV's cv2.Rodrigues)."""
    r = np.asarray(rvec, dtype=np.float64).reshape(3)
    theta = float(np.linalg.norm(r))
    if theta < 1e-12:
        return np.eye(3)
    k = r / theta
    kx = np.array([[0, -k[2], k[1]], [k[2], 0, -k[0]], [-k[1], k[0], 0]])
    return np.eye(3) + math.sin(theta) * kx + (1 - math.cos(theta)) * kx @ kx


def _numbers(text):
    values = []
    for token in (text or '').split():
        values.append(float(token))  # raises ValueError for non-numeric tokens
    return values


def _problem(camera, file, field, reason):
    return {'camera': camera, 'file': file, 'field': field, 'reason': reason}


def _parse_intrinsic(camera, name, data, problems):
    try:
        root = ET.fromstring(data)
    except ET.ParseError as e:
        problems.append(_problem(camera, name, None, f'not valid XML ({e})'))
        return None
    node = root.find('camera_matrix')
    if node is None:
        problems.append(_problem(camera, name, 'camera_matrix', 'missing camera_matrix'))
        return None
    try:
        rows, cols = int(node.findtext('rows')), int(node.findtext('cols'))
        values = _numbers(node.findtext('data'))
    except (TypeError, ValueError):
        problems.append(_problem(camera, name, 'camera_matrix', 'rows, cols or data are not numbers'))
        return None
    if (rows, cols) != (3, 3) or len(values) != 9:
        problems.append(_problem(camera, name, 'camera_matrix',
                                 f'must be 3x3 with 9 values (found {rows}x{cols} with {len(values)} values)'))
        return None
    if not all(math.isfinite(v) for v in values):
        problems.append(_problem(camera, name, 'camera_matrix', 'contains a non-finite value'))
        return None
    return [values[0:3], values[3:6], values[6:9]]


def _parse_vector(camera, name, root, field, problems):
    node = root.find(field)
    if node is None:
        problems.append(_problem(camera, name, field, f'missing {field}'))
        return None
    try:
        values = _numbers(node.text)
    except ValueError:
        problems.append(_problem(camera, name, field, 'contains a value that is not a number'))
        return None
    if len(values) != 3:
        problems.append(_problem(camera, name, field, f'must have 3 values (found {len(values)})'))
        return None
    if not all(math.isfinite(v) for v in values):
        problems.append(_problem(camera, name, field, 'contains a non-finite value'))
        return None
    return values


def _parse_extrinsic(camera, name, data, problems):
    try:
        root = ET.fromstring(data)
    except ET.ParseError as e:
        problems.append(_problem(camera, name, None, f'not valid XML ({e})'))
        return None, None
    return (_parse_vector(camera, name, root, 'rvec', problems),
            _parse_vector(camera, name, root, 'tvec', problems))


def validate_bundle(file_bytes):
    """
    UC-09. Returns (cameras, manifest, problems). `cameras` is a list in the configured camera order with
    intrinsicMatrix (3x3), rotation (rvec) and translationCm (tvec); it is only meaningful when problems is empty.
    """
    problems = []
    try:
        archive = zipfile.ZipFile(io.BytesIO(file_bytes))
    except zipfile.BadZipFile:
        return [], {}, [_problem(None, None, None, 'the bundle is not a valid .zip file')]

    members = {}
    for info in archive.infolist():
        if info.is_dir():
            continue
        name = info.filename.replace('\\', '/')
        parts = name.split('/')
        if '__MACOSX' in parts or parts[-1].startswith('.'):
            continue
        members[name] = info
    # accept one top-level folder around the layout
    prefixes = {n.split('/')[0] + '/' for n in members if '/' in n}
    prefix = ''
    if 'manifest.json' not in members and len(prefixes) == 1:
        prefix = prefixes.pop()

    def read(name):
        info = members.get(prefix + name)
        if info is None:
            return None
        if info.file_size > MAX_MEMBER_BYTES:
            problems.append(_problem(None, name, None, 'file is larger than 1 MB'))
            return None
        return archive.read(info)

    manifest = {}
    raw = read('manifest.json')
    if raw is None:
        problems.append(_problem(None, 'manifest.json', None, 'missing manifest.json'))
    else:
        try:
            manifest = json.loads(raw.decode('utf-8'))
            if not isinstance(manifest, dict):
                raise ValueError
        except (ValueError, UnicodeDecodeError):
            problems.append(_problem(None, 'manifest.json', None, 'manifest.json is not a JSON object'))
            manifest = {}
    if manifest or raw is not None:
        convention = manifest.get('coordinateConvention')
        unit = manifest.get('translationUnit')
        if convention != scene.COORDINATE_CONVENTION:
            problems.append(_problem(None, 'manifest.json', 'coordinateConvention',
                                     f'must be "{scene.COORDINATE_CONVENTION}" (found {json.dumps(convention)})'))
        if unit != scene.TRANSLATION_UNIT:
            problems.append(_problem(None, 'manifest.json', 'translationUnit',
                                     f'must be "{scene.TRANSLATION_UNIT}" (found {json.dumps(unit)})'))
        # Optional per-camera declarations must agree with the bundle-level ones
        for camera, declared in (manifest.get('cameras') or {}).items():
            if not isinstance(declared, dict):
                continue
            for key in ('coordinateConvention', 'translationUnit'):
                if key in declared and declared[key] != manifest.get(key):
                    problems.append(_problem(camera, 'manifest.json', key,
                                             f'{key} differs from the rest of the bundle ({declared[key]})'))

    cameras = []
    for order, camera in enumerate(scene.CAMERA_ORDER):
        intr_name, extr_name = intrinsic_name(camera), extrinsic_name(camera)
        intr_data, extr_data = read(intr_name), read(extr_name)
        if intr_data is None:
            problems.append(_problem(camera, intr_name, None, 'intrinsic calibration file is missing'))
        if extr_data is None:
            problems.append(_problem(camera, extr_name, None, 'extrinsic calibration file is missing'))
        K = _parse_intrinsic(camera, intr_name, intr_data, problems) if intr_data is not None else None
        rvec, tvec = _parse_extrinsic(camera, extr_name, extr_data, problems) if extr_data is not None \
            else (None, None)
        cameras.append({'cameraId': camera, 'orderIndex': order, 'intrinsicMatrix': K, 'rotation': rvec,
                        'translationCm': tvec})
    return cameras, manifest, problems


def bundle_digest(file_bytes):
    return hashlib.sha256(file_bytes).hexdigest()


def projection_matrices(calibration):
    """{cameraId: 3x4 numpy projection matrix} from a stored calibration document."""
    out = {}
    for cam in calibration['cameras']:
        K = np.asarray(cam['intrinsicMatrix'], dtype=np.float64)
        Rt = np.hstack([rodrigues(cam['rotation']), np.asarray(cam['translationCm'], dtype=np.float64)[:, None]])
        out[cam['cameraId']] = K @ Rt
    return out


def project_ground_point(P, x_cm, y_cm):
    """Pixel position of a ground point (z = 0) in a 1920x1080 camera image, or None when not visible."""
    u, v, w = P @ np.array([x_cm, y_cm, 0.0, 1.0])
    if w <= 1e-6:
        return None
    u, v = u / w, v / w
    if not (0 <= u < scene.IMAGE_WIDTH and 0 <= v < scene.IMAGE_HEIGHT):
        return None
    return float(u), float(v)
