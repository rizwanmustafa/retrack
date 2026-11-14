"""
R4 tracker (research-methodology-5/METHODOLOGY.md). Online, one frame at a time:

    positions = tracker.query_positions()          # ref coordinates of the live tracks -> network input
    out = model(item, track_xy=..., ...)            # network predicts each track's new position and presence
    hyps = tracker.update(dets, scores, pred_xy, presence)

Association follows the baseline tracker (tracking/multitracker.py JDETracker) as closely as possible, so that the
comparison isolates the track queries: candidate detections score > thr - 0.1, new tracks only from detections with
score >= thr, a new track is reported from its second matched frame (except in the first frame), the gate is 75
reference cells (1.875 m) and tracks are dropped after `track_buffer` frames without a match. The difference is what
a detection is compared with: the baseline compares the detection's own guess of its previous position with the
track's Kalman prediction; here the track's position predicted by the network from the current frame is compared
with the detection. A track that is not matched but that the network still believes present (presence >= 0.5) moves
to the predicted position, so it can be picked up again later; it is not reported in that frame.
"""
import numpy as np
from scipy.optimize import linear_sum_assignment


class QueryTracker:
    def __init__(self, conf_thres=0.5, track_buffer=5, match_thresh=75., presence_thres=0.5):
        self.conf = conf_thres
        self.buffer = track_buffer
        self.gate = match_thresh
        self.presence_thres = presence_thres
        self.tracks = []  # dicts: id, xy (ref, 2), lost, confirmed
        self.next_id = 1
        self.frame = 0

    def query_positions(self):
        return np.array([t['xy'] for t in self.tracks], dtype=np.float64).reshape(-1, 2)

    def update(self, dets, scores, pred_xy, presence):
        """dets K,2 and scores K (ref coordinates), pred_xy T,2 and presence T for the tracks of query_positions().
        Returns the reported hypotheses [(track_id, xy)]."""
        self.frame += 1
        dets, scores = np.asarray(dets, np.float64).reshape(-1, 2), np.asarray(scores, np.float64).reshape(-1)
        keep = scores > self.conf - 0.1
        dets, scores = dets[keep], scores[keep]
        pred_xy = np.asarray(pred_xy, np.float64).reshape(-1, 2)
        presence = np.asarray(presence, np.float64).reshape(-1)
        assert len(pred_xy) == len(self.tracks) == len(presence)

        matched_t, matched_d = set(), set()
        if len(self.tracks) and len(dets):
            dist = np.linalg.norm(pred_xy[:, None] - dets[None], axis=2)
            cost = np.where(dist <= self.gate, dist, 1e6)
            r, c = linear_sum_assignment(cost)
            for i, j in zip(r, c):
                if cost[i, j] < 1e5:
                    matched_t.add(int(i))
                    matched_d.add(int(j))
                    t = self.tracks[i]
                    t['xy'], t['lost'] = dets[j], 0
                    t['reported'] = True
                    if not t['confirmed']:
                        t['confirmed'] = True
        alive = []
        for i, t in enumerate(self.tracks):
            if i in matched_t:
                alive.append(t)
                continue
            t['reported'] = False
            if not t['confirmed']:
                continue  # unconfirmed tracks that are not matched are dropped (as the baseline)
            t['lost'] += 1
            if presence[i] >= self.presence_thres:
                t['xy'] = pred_xy[i]
            if t['lost'] <= self.buffer:
                alive.append(t)
        for j in range(len(dets)):
            if j in matched_d or scores[j] < self.conf:
                continue
            first = self.frame == 1
            alive.append({'id': self.next_id, 'xy': dets[j], 'lost': 0, 'confirmed': first, 'reported': first})
            self.next_id += 1
        self.tracks = alive
        return [(t['id'], np.asarray(t['xy'], np.float64)) for t in self.tracks if t['reported'] and t['confirmed']]
