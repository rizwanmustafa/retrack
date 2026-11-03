"""
Camera rig client for LIVE sessions: pushes synchronized frame sets from a WILDTRACK-style folder to the ingest
endpoint (docs/LIVE_INGEST.md). Useful for demonstrations and acceptance tests of the LIVE contract.

    python scripts/push_framesets.py --url http://localhost:5000 --session S-000004 --token <ingest token> \
        --images /data/Wildtrack/Image_subsets --start-time 2026-11-03T10:00:00Z --frame-rate 10 --rate 2

Only the Python standard library is used. Each response (accepted or rejected with a reason) is printed.
"""
import argparse
import json
import os
import re
import sys
import time
import urllib.error
import urllib.request
import uuid
from datetime import datetime, timedelta

CAMERAS = ('CVLab1', 'CVLab2', 'CVLab3', 'CVLab4', 'IDIAP1', 'IDIAP2', 'IDIAP3')
FRAME_FILE = re.compile(r'^(\d+)\.(png|jpg|jpeg)$', re.IGNORECASE)


def multipart(fields, files):
    boundary = uuid.uuid4().hex
    body = bytearray()
    for name, value in fields:
        body += (f'--{boundary}\r\nContent-Disposition: form-data; name="{name}"\r\n\r\n{value}\r\n').encode()
    for name, filename, data in files:
        body += (f'--{boundary}\r\nContent-Disposition: form-data; name="{name}"; filename="{filename}"\r\n'
                 'Content-Type: application/octet-stream\r\n\r\n').encode()
        body += data + b'\r\n'
    body += f'--{boundary}--\r\n'.encode()
    return bytes(body), f'multipart/form-data; boundary={boundary}'


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('--url', required=True, help='base URL of the Retrack server')
    p.add_argument('--session', required=True)
    p.add_argument('--token', required=True, help='ingest token shown on the live view')
    p.add_argument('--images', required=True, help='folder with one sub-folder per camera (C1..C7)')
    p.add_argument('--start-time', required=True, help='source time of frame 0, ISO 8601 with offset')
    p.add_argument('--frame-rate', type=float, default=10.0, help='source frame rate used for sourceTime')
    p.add_argument('--rate', type=float, default=2.0, help='frame sets sent per second (0 = as fast as possible)')
    p.add_argument('--first', type=int, default=None)
    p.add_argument('--last', type=int, default=None)
    args = p.parse_args()

    start = datetime.fromisoformat(args.start_time.replace('Z', '+00:00'))
    folders = {cam: os.path.join(args.images, f'C{i + 1}') for i, cam in enumerate(CAMERAS)}
    frames = sorted({int(m.group(1)) for f in os.listdir(folders['CVLab1']) if (m := FRAME_FILE.match(f))})
    frames = [f for f in frames if (args.first is None or f >= args.first) and (args.last is None or f <= args.last)]
    url = f'{args.url.rstrip("/")}/api/ingest/sessions/{args.session}/framesets'

    for frame in frames:
        t0 = time.monotonic()
        source_time = (start + timedelta(seconds=frame / args.frame_rate)).isoformat()
        files = []
        for cam in CAMERAS:
            names = [n for n in os.listdir(folders[cam]) if (m := FRAME_FILE.match(n)) and int(m.group(1)) == frame]
            for name in names[:1]:
                with open(os.path.join(folders[cam], name), 'rb') as f:
                    files.append((cam, name, f.read()))
        body, content_type = multipart([('frameIndex', frame), ('sourceTime', source_time)], files)
        req = urllib.request.Request(url, data=body, method='POST', headers={
            'Content-Type': content_type, 'Authorization': f'Bearer {args.token}'})
        try:
            with urllib.request.urlopen(req) as resp:
                print(frame, resp.status, resp.read().decode())
        except urllib.error.HTTPError as e:
            print(frame, e.code, e.read().decode())
            if e.code in (401, 409):
                sys.exit('the session is not accepting input; stopping')
        if args.rate > 0:
            time.sleep(max(0.0, 1.0 / args.rate - (time.monotonic() - t0)))


if __name__ == '__main__':
    main()
