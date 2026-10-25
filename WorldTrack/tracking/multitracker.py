from collections import OrderedDict
from collections import deque
from typing import List

import numpy as np

from tracking import matching
from tracking.kalman_filter import KalmanFilter


class TrackState(object):
    New = 0
    Tracked = 1
    Lost = 2
    Removed = 3


class BaseTrack(object):
    _count = 0

    track_id = 0
    is_activated = False
    state = TrackState.New

    history = OrderedDict()
    features = []
    curr_feature = None
    score = 0
    start_frame = 0
    frame_id = 0
    time_since_update = 0

    # multi-camera
    location = (np.inf, np.inf)

    @property
    def end_frame(self):
        return self.frame_id

    @staticmethod
    def next_id():
        BaseTrack._count += 1
        return BaseTrack._count

    def activate(self, *args):
        raise NotImplementedError

    def predict(self):
        raise NotImplementedError

    def update(self, *args, **kwargs):
        raise NotImplementedError

    def mark_lost(self):
        self.state = TrackState.Lost

    def mark_removed(self):
        self.state = TrackState.Removed


class STrack(BaseTrack):
    shared_kalman = KalmanFilter()

    def __init__(self, xy, xy_prev, score, buffer_size=30, feat=None):

        # wait activate
        self._xy = xy
        self._xy_prev = xy_prev
        self.kalman_filter = None
        self.mean, self.covariance = None, None
        self.is_activated = False

        self.score = score
        self.tracklet_len = 0

        self.smooth_feat = None
        self.curr_feat = None
        self.features = deque([], maxlen=buffer_size)
        self.alpha = 0.9
        if feat is not None:
            self.update_features(feat)

    def update_features(self, feat):
        feat = np.array(feat, dtype=np.float32)
        feat /= np.linalg.norm(feat)
        self.curr_feat = feat
        if self.smooth_feat is None:
            self.smooth_feat = feat
        else:
            self.smooth_feat = self.alpha * self.smooth_feat + (1 - self.alpha) * feat
        self.features.append(feat)
        self.smooth_feat /= np.linalg.norm(self.smooth_feat)

    def predict(self):
        mean_state = self.mean.copy()
        self.mean, self.covariance = self.kalman_filter.predict(mean_state, self.covariance)

    @staticmethod
    def multi_predict(stracks):
        if len(stracks) > 0:
            multi_mean = np.asarray([st.mean.copy() for st in stracks])
            multi_covariance = np.asarray([st.covariance for st in stracks])
            multi_mean, multi_covariance = STrack.shared_kalman.multi_predict(multi_mean, multi_covariance)
            for i, (mean, cov) in enumerate(zip(multi_mean, multi_covariance)):
                stracks[i].mean = mean
                stracks[i].covariance = cov

    def activate(self, kalman_filter, frame_id):
        """Start a new tracklet"""
        self.kalman_filter = kalman_filter
        self.track_id = self.next_id()
        self.mean, self.covariance = self.kalman_filter.initiate(self.xy)

        self.tracklet_len = 0
        self.state = TrackState.Tracked
        if frame_id == 1:
            self.is_activated = True
        # self.is_activated = True
        self.frame_id = frame_id
        self.start_frame = frame_id

    def re_activate(self, new_track, frame_id, new_id=False):
        self.mean, self.covariance = self.kalman_filter.update(
            self.mean, self.covariance, new_track.xy
        )

        # self.update_features(new_track.curr_feat)
        self.tracklet_len = 0
        self.state = TrackState.Tracked
        self.is_activated = True
        self.frame_id = frame_id
        if new_id:
            self.track_id = self.next_id()

    def update(self, new_track, frame_id, update_feature=False):
        """
        Update a matched track
        :type new_track: STrack
        :type frame_id: int
        :type update_feature: bool
        :return:
        """
        self.frame_id = frame_id
        self.tracklet_len += 1

        self.mean, self.covariance = self.kalman_filter.update(self.mean, self.covariance, new_track.xy)
        self.state = TrackState.Tracked
        self.is_activated = True

        self.score = new_track.score
        if update_feature:
            self.update_features(new_track.curr_feat)

    @property
    def xy(self):
        # if self.state == TrackState.Lost:
        #     return self.mean[:2]
        if self.mean is None:
            return self._xy
        return self.mean[:2]

    @property
    def xy_prev(self):
        return self._xy_prev

    def __repr__(self):
        return 'OT_{}_({}-{})'.format(self.track_id, self.start_frame, self.end_frame)


class JDETracker:
    def __init__(self, conf_thres=0.1, track_buffer=5):
        self.tracked_stracks: List[STrack] = []
        self.lost_stracks: List[STrack] = []
        self.removed_stracks: List[STrack] = []

        self.frame_id = 0
        self.det_thresh = conf_thres
        self.max_time_lost = track_buffer

        self.kalman_filter = KalmanFilter()

    def update(self, dets, dets_prev, score):
        self.frame_id += 1
        activated_starcks = []
        refind_stracks = []
        lost_stracks = []
        removed_stracks = []

        remain_inds = score > self.det_thresh - 0.1
        dets = dets[remain_inds]
        dets_prev = dets_prev[remain_inds]
        # id_feature = id_feature[remain_inds]

        if len(dets) > 0:
            """Detections"""
            detections = [STrack(xy, xy_prev, s, self.max_time_lost) for
                          (xy, xy_prev, s) in zip(dets, dets_prev, score)]
        else:
            detections = []

        """ Add newly detected tracklets to tracked_stracks"""
        unconfirmed = []
        tracked_stracks: List[STrack] = []
        for track in self.tracked_stracks:
            if not track.is_activated:
                unconfirmed.append(track)
            else:
                tracked_stracks.append(track)

        ''' Step 2:association'''
        strack_pool = joint_stracks(tracked_stracks, self.lost_stracks)
        # Predict the current location with KF
        STrack.multi_predict(strack_pool)

        strack_pool_xy = [track.xy for track in strack_pool]
        detections_xy_prev = [det.xy_prev for det in detections]

        dists = matching.center_distance(strack_pool_xy, detections_xy_prev)
        matches, u_track, u_detection = matching.linear_assignment(dists, thresh=75)

        for itracked, idet in matches:
            track = strack_pool[itracked]
            det = detections[idet]
            if track.state == TrackState.Tracked:
                track.update(det, self.frame_id)
                activated_starcks.append(track)
            else:
                track.re_activate(det, self.frame_id, new_id=False)
                refind_stracks.append(track)

        for it in u_track:
            track = strack_pool[it]
            if not track.state == TrackState.Lost:
                track.mark_lost()
                lost_stracks.append(track)

        '''Deal with unconfirmed tracks, usually tracks with only one beginning frame'''
        detections = [detections[i] for i in u_detection]
        detections_xy = [det.xy_prev for det in detections]
        unconfirmed_xy = [track.xy for track in unconfirmed]
        # dists = matching.iou_distance(unconfirmed, detections)
        dists = matching.center_distance(unconfirmed_xy, detections_xy)
        matches, u_unconfirmed, u_detection = matching.linear_assignment(dists, thresh=100)
        for itracked, idet in matches:
            unconfirmed[itracked].update(detections[idet], self.frame_id)
            activated_starcks.append(unconfirmed[itracked])
        for it in u_unconfirmed:
            track = unconfirmed[it]
            track.mark_removed()
            removed_stracks.append(track)

        """ Step 3: Init new stracks"""
        for inew in u_detection:
            track = detections[inew]
            if track.score < self.det_thresh:
                continue
            track.activate(self.kalman_filter, self.frame_id)
            activated_starcks.append(track)
        """ Step 5: Update state"""
        for track in self.lost_stracks:
            if self.frame_id - track.end_frame > self.max_time_lost:
                track.mark_removed()
                removed_stracks.append(track)

        self.tracked_stracks = [t for t in self.tracked_stracks if t.state == TrackState.Tracked]
        self.tracked_stracks = joint_stracks(self.tracked_stracks, activated_starcks)
        self.tracked_stracks = joint_stracks(self.tracked_stracks, refind_stracks)
        self.lost_stracks = sub_stracks(self.lost_stracks, self.tracked_stracks)
        self.lost_stracks.extend(lost_stracks)
        self.lost_stracks = sub_stracks(self.lost_stracks, self.removed_stracks)
        self.removed_stracks.extend(removed_stracks)
        self.tracked_stracks, self.lost_stracks = remove_duplicate_stracks(self.tracked_stracks, self.lost_stracks)
        # get scores of lost tracks
        output_stracks = [track for track in self.tracked_stracks if track.is_activated]

        # print('===========Frame {}=========='.format(self.frame_id))
        # print('Activated: {}'.format([track.track_id for track in activated_starcks]))
        # print('Refind: {}'.format([track.track_id for track in refind_stracks]))
        # print('Lost: {}'.format([track.track_id for track in lost_stracks]))
        # print('Removed: {}'.format([track.track_id for track in removed_stracks]))

        return output_stracks


def joint_stracks(tlista, tlistb):
    exists = {}
    res = []
    for t in tlista:
        exists[t.track_id] = 1
        res.append(t)
    for t in tlistb:
        tid = t.track_id
        if not exists.get(tid, 0):
            exists[tid] = 1
            res.append(t)
    return res


def sub_stracks(tlista, tlistb):
    stracks = {}
    for t in tlista:
        stracks[t.track_id] = t
    for t in tlistb:
        tid = t.track_id
        if stracks.get(tid, 0):
            del stracks[tid]
    return list(stracks.values())


def remove_duplicate_stracks(stracksa, stracksb):
    track_a = [t.xy_prev for t in stracksa]
    track_b = [t.xy for t in stracksb]
    pdist = matching.center_distance(track_a, track_b)
    pairs = np.where(pdist < 6)
    dupa, dupb = list(), list()
    for p, q in zip(*pairs):
        timep = stracksa[p].frame_id - stracksa[p].start_frame
        timeq = stracksb[q].frame_id - stracksb[q].start_frame
        if timep > timeq:
            dupb.append(q)
        else:
            dupa.append(p)
    resa = [t for i, t in enumerate(stracksa) if not i in dupa]
    resb = [t for i, t in enumerate(stracksb) if not i in dupb]
    return resa, resb


class ByteBEVTracker(JDETracker):
    """
    Retrack Stage 2: ByteTrack-style association on BEV points (+ Stage 3: optional appearance cost).

    Pass 1: high-confidence detections (score >= conf_thres) are matched to all tracks (tracked + lost).
            Cost = ground-plane distance between the track's predicted position and the detection's learned
            previous-frame position (as in the baseline). With re-ID it becomes
            D = lambda * D_app + (1 - lambda) * D_motion / gate, restricted to pairs inside the motion gate.
    Pass 2: low-confidence detections (low_thres <= score < conf_thres) are matched, motion only, to the tracks
            left unmatched. They can continue a track but never start one; unmatched ones are dropped.
    New tracks are started only from unmatched high-confidence detections. Lost tracks are kept for
    `track_buffer` frames.
    """

    def __init__(self, conf_thres=0.5, low_thres=0.1, track_buffer=10, match_thresh=75., low_match_thresh=50.,
                 unconfirmed_thresh=100., use_reid=False, reid_lambda=0.98, reid_thresh=0.6, lost_gate_growth=0.0):
        super().__init__(conf_thres=conf_thres, track_buffer=track_buffer)
        self.low_thresh = low_thres
        self.match_thresh = match_thresh
        self.low_match_thresh = low_match_thresh
        self.unconfirmed_thresh = unconfirmed_thresh
        self.use_reid = use_reid
        self.reid_lambda = reid_lambda
        self.reid_thresh = reid_thresh
        self.lost_gate_growth = lost_gate_growth

    def first_pass_cost(self, tracks, detections):
        motion = matching.center_distance([t.xy for t in tracks], [d.xy_prev for d in detections])
        if not self.use_reid or motion.size == 0:
            return motion, self.match_thresh

        # motion gate, optionally widened for tracks that have been lost for a while
        frames_lost = np.array([max(self.frame_id - t.end_frame - 1, 0) for t in tracks], dtype=np.float64)
        gate = self.match_thresh * (1 + self.lost_gate_growth * frames_lost)[:, None]
        motion_norm = motion / gate

        app = motion_norm.copy()  # fall back to motion where a descriptor is missing
        t_ok = np.array([t.smooth_feat is not None for t in tracks])
        d_ok = np.array([d.curr_feat is not None for d in detections])
        if t_ok.any() and d_ok.any():
            t_feat = np.stack([t.smooth_feat for t in tracks if t.smooth_feat is not None])
            d_feat = np.stack([d.curr_feat for d in detections if d.curr_feat is not None])
            cos = np.maximum(0.0, 1.0 - t_feat @ d_feat.T)
            app[np.ix_(t_ok, d_ok)] = cos

        cost = self.reid_lambda * app + (1 - self.reid_lambda) * motion_norm
        cost[motion_norm > 1] = 1e5
        return cost, self.reid_thresh

    def update(self, dets, dets_prev, score, feats=None):
        self.frame_id += 1
        activated_starcks = []
        refind_stracks = []
        lost_stracks = []
        removed_stracks = []

        dets, dets_prev, score = np.asarray(dets), np.asarray(dets_prev), np.asarray(score)
        if feats is None:
            feats = [None] * len(dets)
        keep = np.where(score >= self.low_thresh)[0]
        detections = []
        for i in keep:
            feat = feats[i]
            if feat is not None and not np.all(np.isfinite(feat)):
                feat = None
            detections.append(STrack(dets[i], dets_prev[i], score[i], self.max_time_lost, feat=feat))
        dets_high = [d for d in detections if d.score >= self.det_thresh]
        dets_low = [d for d in detections if d.score < self.det_thresh]

        unconfirmed = []
        tracked_stracks: List[STrack] = []
        for track in self.tracked_stracks:
            if not track.is_activated:
                unconfirmed.append(track)
            else:
                tracked_stracks.append(track)

        def apply_match(track, det):
            if track.state == TrackState.Tracked:
                track.update(det, self.frame_id)
                activated_starcks.append(track)
            else:
                track.re_activate(det, self.frame_id, new_id=False)
                refind_stracks.append(track)
            if det.curr_feat is not None:
                track.update_features(det.curr_feat)

        ''' Pass 1: high-confidence detections vs all tracks'''
        strack_pool = joint_stracks(tracked_stracks, self.lost_stracks)
        STrack.multi_predict(strack_pool)
        dists, thresh = self.first_pass_cost(strack_pool, dets_high)
        matches, u_track, u_detection = matching.linear_assignment(dists, thresh=thresh)
        for itracked, idet in matches:
            apply_match(strack_pool[itracked], dets_high[idet])

        ''' Pass 2: low-confidence detections vs the remaining tracks (motion only)'''
        r_stracks = [strack_pool[i] for i in u_track]
        dists = matching.center_distance([t.xy for t in r_stracks], [d.xy_prev for d in dets_low])
        matches, u_track, _ = matching.linear_assignment(dists, thresh=self.low_match_thresh)
        for itracked, idet in matches:
            apply_match(r_stracks[itracked], dets_low[idet])

        for it in u_track:
            track = r_stracks[it]
            if not track.state == TrackState.Lost:
                track.mark_lost()
                lost_stracks.append(track)

        '''Unconfirmed tracks (one beginning frame) vs remaining high-confidence detections'''
        detections = [dets_high[i] for i in u_detection]
        dists = matching.center_distance([t.xy for t in unconfirmed], [d.xy_prev for d in detections])
        matches, u_unconfirmed, u_detection = matching.linear_assignment(dists, thresh=self.unconfirmed_thresh)
        for itracked, idet in matches:
            unconfirmed[itracked].update(detections[idet], self.frame_id)
            if detections[idet].curr_feat is not None:
                unconfirmed[itracked].update_features(detections[idet].curr_feat)
            activated_starcks.append(unconfirmed[itracked])
        for it in u_unconfirmed:
            track = unconfirmed[it]
            track.mark_removed()
            removed_stracks.append(track)

        """ Init new stracks from high-confidence detections only"""
        for inew in u_detection:
            track = detections[inew]
            track.activate(self.kalman_filter, self.frame_id)
            activated_starcks.append(track)

        for track in self.lost_stracks:
            if self.frame_id - track.end_frame > self.max_time_lost:
                track.mark_removed()
                removed_stracks.append(track)

        self.tracked_stracks = [t for t in self.tracked_stracks if t.state == TrackState.Tracked]
        self.tracked_stracks = joint_stracks(self.tracked_stracks, activated_starcks)
        self.tracked_stracks = joint_stracks(self.tracked_stracks, refind_stracks)
        self.lost_stracks = sub_stracks(self.lost_stracks, self.tracked_stracks)
        self.lost_stracks.extend(lost_stracks)
        self.lost_stracks = sub_stracks(self.lost_stracks, self.removed_stracks)
        self.removed_stracks.extend(removed_stracks)
        self.tracked_stracks, self.lost_stracks = remove_duplicate_stracks(self.tracked_stracks, self.lost_stracks)

        return [track for track in self.tracked_stracks if track.is_activated]


def build_tracker(tracker='baseline', conf_thres=0.5, track_buffer=5, low_thres=0.1, low_match_thresh=50.,
                  use_reid=False, reid_lambda=0.98, reid_thresh=0.6, lost_gate_growth=0.0):
    if tracker == 'baseline':
        return JDETracker(conf_thres=conf_thres, track_buffer=track_buffer)
    if tracker == 'byte':
        return ByteBEVTracker(conf_thres=conf_thres, low_thres=low_thres, track_buffer=track_buffer,
                              low_match_thresh=low_match_thresh, use_reid=use_reid, reid_lambda=reid_lambda,
                              reid_thresh=reid_thresh, lost_gate_growth=lost_gate_growth)
    raise ValueError(f'Unknown tracker {tracker}')
