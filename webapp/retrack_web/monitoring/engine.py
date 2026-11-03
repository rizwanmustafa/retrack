"""
Monitoring sessions (FR-13 to FR-24, UC-11 to UC-13, C-06, NFR-07, NFR-11).

One model instance serves one active analysis session at a time. Start creates the session as Initializing and a
worker thread validates, loads the checkpoint and resets tracking before the session becomes Running; only then is
input accepted. The worker takes frame sets from the session's MultiCameraFeed in source order, runs the AIModel,
commits each processed frame set and publishes it to the in-memory live view. Stop, end of a recorded source or a
fatal error finalizes the session, after which a report is generated when committed results exist.

Viewing is optional: the worker never waits for a viewer, and camera thumbnails are only encoded while someone is
watching. Raw images are never written to storage (NFR-08).
"""
import base64
import hashlib
import hmac
import io
import os
import secrets
import threading
import time
from collections import deque

from PIL import Image

from .. import audit, scene
from ..ai import BackendUnavailable, ModelError
from ..calibration import project_ground_point, projection_matrices
from ..db import get_db, next_id
from ..errors import AppError
from ..reports import service as reports
from ..reports.summary import compute_summary, frame_rows
from ..security import END_USER, Principal
from ..services.calibrations import current_setup, ensure_rig
from ..services.checkpoints import active_checkpoint
from ..textutil import safe_message
from ..timeutil import iso, utcnow
from .feeds import LIVE, RECORDED, FeedClosed, LiveFeed, RecordedFeed, find_recorded_source
from .frameset import MISSING_CAMERA, EndOfSource, FrameSet, InputFault

# Session status
INITIALIZING, RUNNING, COMPLETED, FAILED = 'Initializing', 'Running', 'Completed', 'Failed'
ACTIVE_STATUSES = (INITIALIZING, RUNNING)
# What a Running session is doing right now (NFR-01)
WAITING_FOR_INPUT = 'WAITING_FOR_INPUT'
PROCESSING = 'PROCESSING'
PAUSED_MISSING_CAMERA = 'PAUSED_MISSING_CAMERA'
STOPPING = 'STOPPING'
FINALIZED = 'FINALIZED'
# Why a session ended
END_STOP = 'STOP'
END_OF_SOURCE = 'END_OF_SOURCE'
END_FATAL = 'FATAL_ERROR'
END_INIT = 'INITIALIZATION_FAILED'
END_RESTART = 'SERVER_RESTART'

TRAJECTORY_POINTS = 60
THUMB_SIZE = (480, 270)
VIEWER_IDLE_SECONDS = 15


def _token_hash(token):
    return hashlib.sha256(token.encode()).hexdigest()


class LiveState:
    """The latest published result of a session, kept in memory for the live view."""

    def __init__(self, session, calibration):
        self.lock = threading.Lock()
        self.session_id = session['_id']
        self.runtime = INITIALIZING
        self.message = 'Initializing: validating the configuration and loading the checkpoint.'
        self.frame = None
        self.thumbnails = {}
        self.thumb_frame = None
        self.trajectories = {}
        self.projections = projection_matrices(calibration)
        self.viewer_seen = 0.0


class ActiveSession:
    def __init__(self, session, calibration, checkpoint, feed, token=None):
        self.session_id = session['_id']
        self.user_id = session['userId']
        self.mode = session['inputMode']
        self.calibration = calibration
        self.checkpoint = checkpoint
        self.feed = feed
        self.token = token
        self.token_hash = _token_hash(token) if token else None
        self.stop_event = threading.Event()
        self.running = False
        self.discarded = 0
        self.prev_observed = set()
        self.first_frame = None
        self.thread = None


class MonitoringEngine:
    def __init__(self, app):
        self.app = app
        self._lock = threading.RLock()
        self._active = None
        self._live = {}

    # ------------------------------------------------------------------ state
    def exclusive(self):
        """Held while starting a session or changing the checkpoint, so the two cannot interleave."""
        return self._lock

    def is_busy(self):
        with self._lock:
            if self._active is not None:
                return True
        return get_db(self.app).sessions.find_one({'status': {'$in': list(ACTIVE_STATUSES)}}) is not None

    def active_session_id(self):
        with self._lock:
            return self._active.session_id if self._active else None

    def ingest_token(self, session_id, principal: Principal):
        with self._lock:
            a = self._active
            if a and a.session_id == session_id and a.user_id == principal.id and a.mode == LIVE:
                return a.token
        return None

    # ------------------------------------------------------------------ start / stop
    def start(self, principal: Principal, mode, source_id=None):
        """UC-11 steps 1-2. Returns the new session document; the worker continues in the background."""
        assert principal.role == END_USER
        db = get_db(self.app)
        config = self.app.config
        with self._lock:
            if self.is_busy():
                audit.record(audit.SESSION_START_FAILED, audit.REFUSED, 'Session', None,
                             {'reason': 'model instance busy'})
                raise AppError('The model instance is busy with another monitoring session, so Start was refused. '
                               'Session history and reports remain available.', status=409, code='model_busy')
            problems = []
            setup, calibration = current_setup(principal.id)
            if calibration is None:
                problems.append({'field': 'calibration', 'reason': 'Select or upload a valid calibration bundle.'})
            if mode not in (LIVE, RECORDED):
                problems.append({'field': 'inputMode', 'reason': 'Choose LIVE or RECORDED input.'})
            source = None
            if mode == RECORDED:
                source = find_recorded_source(config['RECORDED_SOURCES_FILE'], source_id or '')
                if source is None:
                    problems.append({'field': 'recordedSource', 'reason': 'Choose a configured recorded source.'})
                elif not os.path.isdir(source['path']):
                    problems.append({'field': 'recordedSource', 'reason': 'This recorded source is not available '
                                                                          'on the server.'})
            checkpoint = active_checkpoint(self.app)
            if checkpoint is None:
                problems.append({'field': 'checkpoint', 'reason': 'No compatible checkpoint is active. Ask the '
                                                                  'Admin to select one.'})
            elif not checkpoint.get('available') or not os.path.isfile(checkpoint['filePath']):
                problems.append({'field': 'checkpoint', 'reason': 'The active checkpoint file is unavailable. Ask '
                                                                  'the Admin to select another one.'})
            if problems:
                audit.record(audit.SESSION_START_FAILED, audit.REFUSED, 'Session', None,
                             {'reason': 'invalid session configuration', 'problems': problems})
                raise AppError('The session configuration is not valid, so the session was not started.',
                               status=422, code='invalid_setup', problems=problems)

            now = utcnow()
            session_id = next_id('sessions', self.app)
            feed_id = next_id('feeds', self.app)
            token = secrets.token_urlsafe(24) if mode == LIVE else None
            session = {
                '_id': session_id,
                'userId': principal.id,
                'username': principal.username,
                'inputMode': mode,
                'feed': {'feedId': feed_id, 'mode': mode, 'status': 'Created',
                         'source': {'id': source['id'], 'name': source['name']} if source else
                         {'id': 'http-push', 'name': 'Camera rig HTTP push'}},
                'rigId': scene.RIG_ID,
                'calibrationId': calibration['_id'],
                'cameraOrder': list(calibration['cameraOrder']),
                'checkpoint': {'checkpointId': checkpoint['_id'], 'fileName': checkpoint['fileName'],
                               'modelName': checkpoint.get('modelName'), 'version': checkpoint.get('version'),
                               'sha256': checkpoint.get('sha256')},
                'status': INITIALIZING,
                'runtimeStatus': INITIALIZING,
                'startTime': now,
                'runningSince': None,
                'endTime': None,
                'endReason': None,
                'stopRequestedAt': None,
                'framesProcessed': 0,
                'framesRejected': 0,
                'framesDiscarded': 0,
                'firstFrameIndex': None,
                'lastFrameIndex': None,
                'lastSourceTime': None,
                'lastInputFault': None,
                'failure': None,
                'summary': None,
                'environment': None,
            }
            db.sessions.insert_one(session)
            ensure_rig(self.app)
            db.camera_rigs.update_one({'_id': scene.RIG_ID}, {'$set': {'calibrationID': calibration['_id']}})
            stop_event = threading.Event()
            if mode == LIVE:
                feed = LiveFeed(feed_id, session_id, config['LIVE_QUEUE_SIZE'])
            else:
                feed = RecordedFeed(feed_id, session_id, source, stop_event)
            active = ActiveSession(session, calibration, checkpoint, feed, token)
            active.stop_event = stop_event
            self._live[session_id] = LiveState(session, calibration)
            self._active = active
            active.thread = threading.Thread(target=self._worker, args=(active,), name=f'session-{session_id}',
                                             daemon=True)
            active.thread.start()
        return session

    def stop(self, session_id, principal: Principal):
        """UC-11 step 3: stops accepting input; finalization continues in the worker."""
        db = get_db(self.app)
        session = db.sessions.find_one({'_id': session_id})
        if session is None or session['userId'] != principal.id:
            raise AppError('This session does not exist or you may not access it.', status=404, code='not_found')
        with self._lock:
            active = self._active if self._active and self._active.session_id == session_id else None
        if active is None:
            if session['status'] in (COMPLETED, FAILED):
                raise AppError(f'This session is already finalized ({session["status"]}).', status=409,
                               code='already_finalized')
            raise AppError('This session is not active on this server.', status=409, code='not_active')
        if not active.stop_event.is_set():
            active.stop_event.set()
            active.discarded += active.feed.stop() or 0
            db.sessions.update_one({'_id': session_id}, {'$set': {'stopRequestedAt': utcnow(),
                                                                  'runtimeStatus': STOPPING}})
            self._set_live(session_id, STOPPING, 'Stop requested: input is closed and the session is being '
                                                 'finalized.')
        return db.sessions.find_one({'_id': session_id})

    # ------------------------------------------------------------------ LIVE input
    def ingest(self, session_id, token, frame_index, source_time, parts):
        """SSD 10.6 supplyFrameSet. Returns ('accepted', FrameSet) or ('rejected', InputFault)."""
        with self._lock:
            active = self._active if self._active and self._active.session_id == session_id else None
        if active is None or active.mode != LIVE:
            raise AppError('This session is not accepting input.', status=409, code='session_not_running')
        if not token or not hmac.compare_digest(_token_hash(token), active.token_hash):
            raise AppError('The ingest token is not valid for this session.', status=401, code='invalid_token')
        if not active.running or active.stop_event.is_set():
            raise AppError('This session is not accepting input.', status=409, code='session_not_running')
        try:
            return 'accepted', active.feed.submit(frame_index, source_time, parts)
        except FeedClosed:
            raise AppError('This session is not accepting input.', status=409, code='session_not_running')
        except InputFault as fault:
            self.record_input_fault(active, fault)
            return 'rejected', fault

    def record_input_fault(self, active, fault):
        """FR-17: an invalid frame set is skipped and recorded; a missing camera pauses inference."""
        db = get_db(self.app)
        info = dict(fault.to_dict(), at=iso(utcnow()))
        db.sessions.update_one({'_id': active.session_id},
                               {'$inc': {'framesRejected': 1}, '$set': {'lastInputFault': info}})
        audit.record(audit.INPUT_ERROR, audit.FAILURE, 'Session', active.session_id, fault.to_dict(),
                     actor=audit.SYSTEM_ACTOR, error=True, app=self.app)
        label = f'frame {fault.frame_index}' if fault.frame_index is not None else 'a frame set'
        if fault.kind == MISSING_CAMERA:
            self._set_runtime(active, PAUSED_MISSING_CAMERA,
                              f'Inference paused: {label} has a {fault.reason}. Waiting for a complete, valid frame '
                              'set. Stop remains available.')
        else:
            with self._live[active.session_id].lock:
                runtime = self._live[active.session_id].runtime
            self._set_runtime(active, runtime, f'Skipped {label}: {fault.reason}.')

    # ------------------------------------------------------------------ worker
    def _worker(self, active):
        with self.app.app_context():
            try:
                self._run(active)
            except Exception as e:
                self.app.logger.exception('Monitoring worker crashed for %s', active.session_id)
                try:
                    self._finalize(active, END_FATAL, {'stage': 'runtime', 'message': safe_message(e)})
                except Exception:
                    self.app.logger.exception('Finalization failed for %s', active.session_id)
            finally:
                with self._lock:
                    if self._active is active:
                        self._active = None

    def _run(self, active):
        db = get_db(self.app)
        model = self.app.extensions['retrack_ai']
        env = {}
        try:
            env = model.describe_environment()
            model.load_checkpoint(active.checkpoint['filePath'])
            model.reset_tracking()
            active.feed.start()
        except (BackendUnavailable, ModelError) as e:
            self._fail_initialization(active, str(e), env)
            return
        except Exception as e:
            self._fail_initialization(active, safe_message(e), env)
            return
        db.sessions.update_one({'_id': active.session_id}, {'$set': {
            'status': RUNNING, 'runtimeStatus': WAITING_FOR_INPUT, 'runningSince': utcnow(), 'environment': env,
            'feed.status': 'Open'}})
        active.running = True
        owner = {'type': END_USER, 'id': active.user_id,
                 'username': db.sessions.find_one({'_id': active.session_id})['username']}
        audit.record(audit.SESSION_STARTED, audit.SUCCESS, 'Session', active.session_id,
                     {'inputMode': active.mode, 'checkpointId': active.checkpoint['_id'],
                      'calibrationId': active.calibration['_id']}, actor=owner, app=self.app)
        self._set_runtime(active, WAITING_FOR_INPUT, self._waiting_message(active))
        reason, failure = self._loop(active, model)
        self._finalize(active, reason, failure)

    def _waiting_message(self, active):
        if active.mode == LIVE:
            return 'Running: waiting for the camera rig to supply a frame set.'
        return 'Running: waiting for the next recorded frame set.'

    def _loop(self, active, model):
        live = self._live[active.session_id]
        while True:
            if active.stop_event.is_set():
                return END_STOP, None
            item = active.feed.get_next_frame_set(timeout=0.5)
            if item is None:
                with live.lock:
                    paused = live.runtime == PAUSED_MISSING_CAMERA
                if not paused:
                    self._set_runtime(active, WAITING_FOR_INPUT, self._waiting_message(active))
                continue
            if isinstance(item, EndOfSource):
                return (END_STOP if active.stop_event.is_set() else END_OF_SOURCE), None
            if isinstance(item, InputFault):
                self.record_input_fault(active, item)
                continue
            assert isinstance(item, FrameSet) and item.validate()
            if active.stop_event.is_set():
                active.discarded += 1
                return END_STOP, None
            self._set_runtime(active, PROCESSING, f'Running: processing frame {item.frame_index}.')
            t0 = time.monotonic()
            try:
                result = model.analyze(item, active.calibration)
            except Exception as e:
                message = str(e) if isinstance(e, ModelError) else safe_message(e)
                return END_FATAL, {'stage': 'inference', 'message': message, 'frameIndex': item.frame_index}
            try:
                self._commit(active, item, result, round((time.monotonic() - t0) * 1000, 1))
            except Exception as e:
                return END_FATAL, {'stage': 'storage', 'message': safe_message(e), 'frameIndex': item.frame_index}
            self._publish(active, item, result)

    def _commit(self, active, frame_set, result, processing_ms):
        """FR-19, FR-20: Track records first, then the frame record that commits the frame set."""
        db = get_db(self.app)
        sid = active.session_id
        obs = [{'trackId': o.track_id, 'x': round(o.x_cm, 1), 'y': round(o.y_cm, 1),
                'confidence': round(o.confidence, 4)} for o in result.observations]
        for o in obs:
            sample = {'frameIndex': frame_set.frame_index, 'timestamp': frame_set.timestamp, 'x': o['x'],
                      'y': o['y'], 'confidence': o['confidence']}
            db.tracks.update_one(
                {'sessionId': sid, 'trackId': o['trackId']},
                {'$setOnInsert': {'firstFrameIndex': frame_set.frame_index, 'firstSeen': frame_set.timestamp},
                 '$set': {'lastFrameIndex': frame_set.frame_index, 'lastSeen': frame_set.timestamp,
                          'groundPosition': {'x': o['x'], 'y': o['y']}, 'confidence': o['confidence'],
                          'status': 'Tracked'},
                 '$push': {'trajectory': sample}},
                upsert=True)
        observed = {o['trackId'] for o in obs}
        lost = active.prev_observed - observed
        if lost:
            db.tracks.update_many({'sessionId': sid, 'trackId': {'$in': sorted(lost)}}, {'$set': {'status': 'Lost'}})
        active.prev_observed = observed
        db.frame_results.insert_one({
            'sessionId': sid, 'frameIndex': frame_set.frame_index, 'timestamp': frame_set.timestamp,
            'observations': obs, 'occupancy': len(obs), 'positionUnit': scene.TRANSLATION_UNIT,
            'receivedAt': frame_set.received_at, 'committedAt': utcnow(), 'processingMs': processing_ms,
        })
        update = {'lastFrameIndex': frame_set.frame_index, 'lastSourceTime': frame_set.timestamp}
        if active.first_frame is None:
            active.first_frame = update['firstFrameIndex'] = frame_set.frame_index
        db.sessions.update_one({'_id': sid}, {'$inc': {'framesProcessed': 1}, '$set': update})

    def _publish(self, active, frame_set, result):
        live = self._live[active.session_id]
        frame = {'frameIndex': frame_set.frame_index, 'timestamp': frame_set.timestamp,
                 'observations': [{'trackId': o.track_id, 'x': round(o.x_cm, 1), 'y': round(o.y_cm, 1),
                                   'confidence': round(o.confidence, 4)} for o in result.observations]}
        watching = time.monotonic() - live.viewer_seen < VIEWER_IDLE_SECONDS
        thumbnails = _thumbnails(frame_set) if watching else None
        with live.lock:
            live.frame = frame
            for o in frame['observations']:
                live.trajectories.setdefault(o['trackId'], deque(maxlen=TRAJECTORY_POINTS)).append((o['x'], o['y']))
            if thumbnails is not None:
                live.thumbnails, live.thumb_frame = thumbnails, frame_set.frame_index
            if live.runtime == PAUSED_MISSING_CAMERA:
                live.runtime = PROCESSING

    def _set_runtime(self, active, runtime, message):
        changed = self._set_live(active.session_id, runtime, message)
        if changed:
            get_db(self.app).sessions.update_one({'_id': active.session_id, 'status': RUNNING},
                                                 {'$set': {'runtimeStatus': runtime}})

    def _set_live(self, session_id, runtime, message):
        live = self._live.get(session_id)
        if live is None:
            return False
        with live.lock:
            if live.runtime == STOPPING and runtime not in (FINALIZED,):
                return False
            changed = live.runtime != runtime
            live.runtime, live.message = runtime, message
        return changed

    # ------------------------------------------------------------------ finalization
    def _fail_initialization(self, active, message, env):
        """FR-13: a failed initialization is recorded and never leaves the session Running."""
        db = get_db(self.app)
        active.feed.stop()
        failure = {'stage': 'initialization', 'message': message, 'at': iso(utcnow())}
        db.sessions.update_one({'_id': active.session_id}, {'$set': {
            'status': FAILED, 'runtimeStatus': FINALIZED, 'endTime': utcnow(), 'endReason': END_INIT,
            'failure': failure, 'environment': env, 'feed.status': 'Closed',
            'summary': compute_summary([])}})
        audit.record(audit.SESSION_START_FAILED, audit.FAILURE, 'Session', active.session_id, failure,
                     actor=audit.SYSTEM_ACTOR, error=True, app=self.app)
        self._set_live(active.session_id, FINALIZED, f'Initialization failed: {message}')

    def _finalize(self, active, reason, failure):
        """FR-23, FR-24: release the feed, close the session and generate the report when results exist."""
        db = get_db(self.app)
        active.discarded += active.feed.stop() or 0
        active.running = False
        self._close_session(active.session_id, reason, failure, discarded=active.discarded)
        status = FAILED if failure else COMPLETED
        session = db.sessions.find_one({'_id': active.session_id})
        if status == COMPLETED:
            message = ('Completed: the recorded source ended.' if reason == END_OF_SOURCE else 'Completed: stopped.')
        else:
            message = f'Failed: {failure["message"]}'
        if not session['framesProcessed']:
            message += ' No frame set was processed, so no report was generated.'
        self._set_live(active.session_id, FINALIZED, message)
        self._auto_report(session)

    def _close_session(self, session_id, reason, failure, discarded=0):
        db = get_db(self.app)
        status = FAILED if failure else COMPLETED
        db.tracks.update_many({'sessionId': session_id}, {'$set': {'status': 'Ended'}})
        summary = compute_summary(frame_rows(session_id, self.app))
        update = {'status': status, 'runtimeStatus': FINALIZED, 'endTime': utcnow(), 'endReason': reason,
                  'framesProcessed': summary['framesProcessed'], 'summary': summary, 'feed.status': 'Closed'}
        if discarded:
            update['framesDiscarded'] = discarded
        if failure:
            update['failure'] = dict(failure, at=iso(utcnow()))
        db.sessions.update_one({'_id': session_id}, {'$set': update})
        if failure:
            audit.record(audit.PROCESSING_ERROR, audit.FAILURE, 'Session', session_id, failure,
                         actor=audit.SYSTEM_ACTOR, error=True, app=self.app)
            audit.record(audit.SESSION_FAILED, audit.FAILURE, 'Session', session_id,
                         dict(summary, endReason=reason), actor=audit.SYSTEM_ACTOR, error=True, app=self.app)
        else:
            audit.record(audit.SESSION_COMPLETED, audit.SUCCESS, 'Session', session_id,
                         dict(summary, endReason=reason), actor=audit.SYSTEM_ACTOR, app=self.app)

    def _auto_report(self, session):
        if not session['framesProcessed']:
            return None
        try:
            return reports.generate(session['_id'], reports.AUTOMATIC, audit.SYSTEM_ACTOR, app=self.app)
        except Exception:
            self.app.logger.exception('Automatic report failed for %s', session['_id'])
            return None

    def recover_interrupted(self):
        """Sessions left active by a server stop are marked Failed; inference is not resumed (NFR-16)."""
        db = get_db(self.app)
        for session in list(db.sessions.find({'status': {'$in': list(ACTIVE_STATUSES)}})):
            self._close_session(session['_id'], END_RESTART, {
                'stage': 'runtime', 'message': 'The server stopped while this session was active. Interrupted '
                                               'inference is not resumed; committed results were kept.'})
            self._auto_report(db.sessions.find_one({'_id': session['_id']}))

    def wait_idle(self, timeout=30):
        """Test helper: waits for the active worker to finish."""
        with self._lock:
            active = self._active
        if active and active.thread:
            active.thread.join(timeout)

    # ------------------------------------------------------------------ live view
    def snapshot(self, session_id, principal: Principal, include_images=False):
        """FR-21, FR-22: the structured tracking result for the owner's session."""
        db = get_db(self.app)
        session = db.sessions.find_one({'_id': session_id})
        if session is None or principal.role != END_USER or session['userId'] != principal.id:
            raise AppError('This session does not exist or you may not access it.', status=404, code='not_found')
        live = self._live.get(session_id)
        thumbnails, thumb_frame = {}, None
        if live is not None:
            with live.lock:
                live.viewer_seen = time.monotonic()
                runtime, message = live.runtime, live.message
                frame = live.frame
                trajectories = {k: list(v) for k, v in live.trajectories.items()}
                if include_images:
                    thumbnails, thumb_frame = dict(live.thumbnails), live.thumb_frame
                projections = live.projections
        else:
            runtime, message = session['runtimeStatus'], _stored_message(session)
            frame, trajectories = _frame_from_db(db, session_id)
            calibration = db.calibrations.find_one({'_id': session['calibrationId']})
            projections = projection_matrices(calibration) if calibration else {}
        if session['status'] in (COMPLETED, FAILED):
            runtime = FINALIZED
            if live is None or not message:
                message = _stored_message(session)

        response = {
            'session': {
                'sessionId': session['_id'], 'inputMode': session['inputMode'], 'status': session['status'],
                'runtimeStatus': runtime, 'statusMessage': message, 'startTime': iso(session['startTime']),
                'endTime': iso(session['endTime']), 'endReason': session['endReason'],
                'framesProcessed': session['framesProcessed'], 'framesRejected': session['framesRejected'],
                'stopRequested': session['stopRequestedAt'] is not None,
                'lastInputFault': session.get('lastInputFault'),
                'summary': session.get('summary') if session['status'] in (COMPLETED, FAILED) else None,
            },
            'model': {'modelId': session['checkpoint'].get('modelName'),
                      'checkpointId': session['checkpoint']['checkpointId'],
                      'version': session['checkpoint'].get('version')},
            'frameSet': None,
            'results': None,
            'scene': {'cameraOrder': session['cameraOrder'], 'groundXRangeCm': list(scene.GROUND_X_RANGE_CM),
                      'groundYRangeCm': list(scene.GROUND_Y_RANGE_CM)},
        }
        if frame is not None:
            response['frameSet'] = {'frameIndex': frame['frameIndex'], 'sourceTime': iso(frame['timestamp']),
                                    'cameraOrder': session['cameraOrder'],
                                    'resolution': {'width': scene.IMAGE_WIDTH, 'height': scene.IMAGE_HEIGHT}}
            tracks = []
            for o in frame['observations']:
                pixels = {}
                for camera, P in projections.items():
                    uv = project_ground_point(P, o['x'], o['y'])
                    if uv is not None:
                        pixels[camera] = {'u': round(uv[0], 1), 'v': round(uv[1], 1)}
                tracks.append({'trackId': o['trackId'], 'groundPosition': {'x': o['x'], 'y': o['y']},
                               'confidence': o['confidence'], 'cameraPixels': pixels})
            observed = {o['trackId'] for o in frame['observations']}
            response['results'] = {
                'units': {'groundPosition': scene.TRANSLATION_UNIT, 'coordinateSystem': scene.COORDINATE_CONVENTION,
                          'cameraPixels': 'pixels in the 1920x1080 source image', 'confidence': 'model score, 0-1'},
                'occupancy': len(frame['observations']),
                'tracks': tracks,
                'trajectories': [{'trackId': k, 'points': [[x, y] for x, y in v]}
                                 for k, v in sorted(trajectories.items()) if k in observed],
            }
        if include_images:
            response['display'] = {'frameIndex': thumb_frame, 'thumbnailSize': list(THUMB_SIZE),
                                   'cameraImages': thumbnails}
        return response


def _thumbnails(frame_set):
    out = {}
    for camera, rgb in frame_set.images.items():
        buf = io.BytesIO()
        Image.fromarray(rgb).resize(THUMB_SIZE, Image.BILINEAR).save(buf, format='JPEG', quality=70)
        out[camera] = 'data:image/jpeg;base64,' + base64.b64encode(buf.getvalue()).decode()
    return out


def _frame_from_db(db, session_id):
    last = db.frame_results.find_one({'sessionId': session_id}, sort=[('frameIndex', -1)])
    if last is None:
        return None, {}
    frame = {'frameIndex': last['frameIndex'], 'timestamp': last['timestamp'], 'observations': last['observations']}
    trajectories = {}
    for track in db.tracks.find({'sessionId': session_id, 'trackId': {'$in': [o['trackId'] for o in
                                                                              last['observations']]}},
                                {'trackId': 1, 'trajectory': {'$slice': -TRAJECTORY_POINTS}}):
        trajectories[track['trackId']] = [(p['x'], p['y']) for p in track.get('trajectory', [])]
    return frame, trajectories


def _stored_message(session):
    status = session['status']
    if status == COMPLETED:
        text = 'Completed: the recorded source ended.' if session['endReason'] == END_OF_SOURCE \
            else 'Completed: stopped.'
    elif status == FAILED:
        text = f'Failed: {(session.get("failure") or {}).get("message", "processing failed")}'
    elif status == INITIALIZING:
        text = 'Initializing: validating the configuration and loading the checkpoint.'
    else:
        text = 'Running.'
    if status in (COMPLETED, FAILED) and not session['framesProcessed']:
        text += ' No frame set was processed, so no report was generated.'
    return text
