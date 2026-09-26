import numpy as np

HOLD_SECONDS = 0.6
GAP_SECONDS = 0.2
SETTLE_SECONDS = 0.8


def is_start_sign(hand, aspect=16 / 9):
    points = np.asarray([(p.x * aspect, p.y, p.z * aspect) for p in hand.landmarks])
    if len(points) != 21 or not np.isfinite(points).all():
        return False
    palm = np.linalg.norm(points[9] - points[0])
    if palm < 1e-6:
        return False

    def distance(a, b):
        return np.linalg.norm(points[a] - points[b]) / palm

    def reach(tip, middle_joint):
        return distance(tip, 0) - distance(middle_joint, 0)

    lower, upper = points[6] - points[5], points[8] - points[6]
    straight = np.dot(lower, upper) / max(np.linalg.norm(lower) * np.linalg.norm(upper), 1e-9)
    index_up = reach(8, 6) > 0.25 and straight > 0.8 and distance(8, 5) > 0.7
    folded = all(reach(tip, joint) < 0.0 for tip, joint in ((12, 10), (16, 14), (20, 18)))
    return index_up and folded and distance(4, 10) < 0.7


class StartSign:
    def __init__(self):
        self.since = None
        self.seen = None

    def update(self, hands, now, aspect):
        if not any(is_start_sign(hand, aspect) for hand in hands):
            return False
        if self.since is None or now - self.seen > GAP_SECONDS:
            self.since = now
        self.seen = now
        if now - self.since >= HOLD_SECONDS:
            self.since = None
            return True
        return False

    def progress(self, now):
        if self.since is None or now - self.seen > GAP_SECONDS:
            return 0.0
        return min((now - self.since) / HOLD_SECONDS, 1.0)
