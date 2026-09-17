import unittest

import numpy as np

from helper_finger_press import FingerPressDetector
from helper_keyboard import KeyboardCalibration, SpacedKeyboardLayout
from helper_motion import MovementModel, MovementRecorder, NO_KEY
from helper_vision import HandSnapshot, Landmark, TrackedHand
from test_helper_motion import example, pose, snapshot


class FingerPressTests(unittest.TestCase):
    def setUp(self):
        self.layout = SpacedKeyboardLayout()
        self.calibration = KeyboardCalibration(((0, 0), (1, 0), (1, 1), (0, 1)))
        self.model = MovementModel(self.layout, {})
        # A 3:1 image keeps the 15 by 5 test keyboard square in pixels.
        self.detector = FingerPressDetector(self.layout, self.calibration, self.model, aspect=3.0)
        self.time = 0

    def feed(self, points=None, hand="Left"):
        self.time += 0.12
        return self.detector.update(snapshot(self.time, points, hand=hand))

    def prepare(self):
        for _ in range(5):
            self.assertEqual(self.feed(), [])

    def test_all_fingers_work_on_either_hand(self):
        for hand in ("Left", "Right"):
            for tip in (8, 12, 16, 20):
                self.detector.reset()
                for _ in range(5):
                    self.feed(hand=hand)
                clicks = []
                for points in example(finger=tip, coupled=None).points:
                    clicks.extend(self.feed(points, hand))
                self.assertEqual(len(clicks), 1, (hand, tip))
                self.assertEqual(clicks[0].tip, tip)

    def test_thumb_types_only_space(self):
        # The resting thumb is over G; a thumb press there must not type.
        clicks = []
        self.prepare()
        for points in example(finger=4, coupled=None).points:
            clicks.extend(self.feed(points))
        self.assertEqual(clicks, [])
        on_space = pose()
        on_space[4] = [7.0, 4.3]
        self.detector.reset()
        for _ in range(5):
            self.feed(on_space)
        for points in example(finger=4, coupled=None).points:
            points[4] = on_space[4] + [0, points[4, 1] - 2.5]
            clicks.extend(self.feed(points))
        self.assertEqual([c.key.name for c in clicks], ["space"])

    def test_same_label_on_both_hands_does_not_disable_typing(self):
        # MediaPipe sometimes calls both hands "Right" when palms are out of view.
        clicks = []
        for index in range(5 + len(example().points)):
            self.time += 0.12
            left = example(coupled=None).points[index - 5] if index >= 5 else pose()
            right = pose() + [7.5, 0]
            hands = snapshot(self.time, left, hand="Right").hands + snapshot(self.time, right, hand="Right", hand_id=2).hands
            clicks.extend(self.detector.update(HandSnapshot(self.time, hands)))
        self.assertEqual([(c.key.name, c.handedness) for c in clicks], [("a", "Left")])

    def test_hand_label_flip_during_a_press_keeps_the_press(self):
        self.prepare()
        clicks = []
        for index, points in enumerate(example(coupled=None).points):
            clicks.extend(self.feed(points, hand="Right" if index % 2 else "Left"))
        self.assertEqual([c.key.name for c in clicks], ["a"])

    def test_slow_press_at_thirty_frames_per_second(self):
        for _ in range(10):
            self.time += 1 / 30
            self.detector.update(snapshot(self.time))
        clicks = []
        # Half a second down and back: a per-frame baseline would absorb it at 30 FPS.
        for progress in np.r_[np.linspace(0, 1, 16), np.linspace(1, 0, 16), np.zeros(6)]:
            self.time += 1 / 30
            points = pose()
            points[20, 1] += 0.45 * progress
            clicks.extend(self.detector.update(snapshot(self.time, points)))
        self.assertEqual([c.key.name for c in clicks], ["a"])

    def test_saved_no_key_examples_do_not_veto_basic_presses(self):
        # Old recordings of resting motion used to swallow small real presses.
        press = [0.45 * np.sin(np.pi * index / 8) if index <= 8 else 0 for index in range(12)]
        for _ in range(3):
            self.model.add(NO_KEY, example(coupled=None, amplitude=0.45))
        self.prepare()
        clicks = []
        for offset in press:
            self.time += 1 / 30
            points = pose()
            points[20, 1] += offset
            clicks.extend(self.detector.update(snapshot(self.time, points)))
        self.assertEqual([c.key.name for c in clicks], ["a"])

    def test_press_size_does_not_depend_on_overlay_shape(self):
        # The same camera motion must count the same after dragging the corners.
        def peak_level(corners):
            detector = FingerPressDetector(self.layout, KeyboardCalibration(corners), self.model)
            image = np.asarray([[0.30, 0.20]] * 21)
            image[5], image[17] = [0.36, 0.25], [0.24, 0.25]
            for tip, x in ((20, 0.245), (16, 0.28), (12, 0.32), (8, 0.355), (4, 0.40)):
                image[tip] = [x, 0.42]
            levels = []
            for index in range(-6, 12):
                frame = image.copy()
                frame[20, 1] += 0.03 * np.sin(np.pi * index / 8) if 0 <= index <= 8 else 0
                hand = TrackedHand(1, "Left", tuple(Landmark(x, y, 0) for x, y in frame))
                detector.update(HandSnapshot(1 + index / 30, (hand,)))
                levels.append(detector.levels.get((1, 20), 0.0))
            return max(levels)
        wide = peak_level(((0.0, 0.0), (1.0, 0.0), (1.0, 1.0), (0.0, 1.0)))
        squashed = peak_level(((0.0, 0.01), (0.99, 0.04), (0.98, 0.77), (0.01, 0.78)))
        self.assertAlmostEqual(wide, squashed, places=6)
        self.assertGreater(wide, 1.0)

    def test_neighbor_need_not_return_before_pinky_can_type(self):
        self.prepare()
        old = MovementRecorder(self.calibration)
        for i in range(5):
            old.update(snapshot(i * 0.12))
        old_events, clicks = [], []
        for index, points in enumerate(example().points):
            if index >= 2:
                points[16, 1] = 2.82
            frame = snapshot(self.time + 0.12, points)
            old_events.extend(old.update(frame))
            clicks.extend(self.feed(points))
        self.assertEqual(old_events, [])
        self.assertEqual([click.key.name for click in clicks], ["a"])

    def test_partial_release_is_enough_without_exact_original_pose(self):
        self.prepare()
        clicks = []
        movement = example(coupled=None)
        for index, points in enumerate(movement.points):
            if index >= 6:
                points[20, 1] = 2.63
            clicks.extend(self.feed(points))
        self.assertEqual([c.key.name for c in clicks], ["a"])

    def test_accompanying_finger_can_rearm_in_its_new_relaxed_pose(self):
        self.prepare()
        clicks = []
        for index, points in enumerate(example().points):
            if index >= 2:
                points[16, 1] = 2.82
            clicks.extend(self.feed(points))
        raised = pose()
        raised[16, 1] = 2.82
        for _ in range(6):
            clicks.extend(self.feed(raised))
        for points in example(finger=16, coupled=None).points:
            points[16, 1] += 0.32
            clicks.extend(self.feed(points))
        self.assertEqual([c.key.name for c in clicks], ["a", "s"])

    def test_equal_simultaneous_fingers_are_both_rejected(self):
        self.prepare()
        clicks = []
        for points in example().points:
            points[16, 1] = points[20, 1]
            clicks.extend(self.feed(points))
        self.assertEqual(clicks, [])

    def test_other_hand_does_not_block_a_press(self):
        self.prepare()
        clicks = []
        for index, points in enumerate(example(coupled=None).points):
            self.time += 0.12
            right = pose() + [0.03 * index, 0]
            hands = snapshot(self.time, points).hands + snapshot(self.time, right, hand="Right", hand_id=2).hands
            clicks.extend(self.detector.update(HandSnapshot(self.time, hands)))
        self.assertEqual([c.key.name for c in clicks], ["a"])

    def test_rest_translation_and_one_frame_noise_do_not_type(self):
        self.prepare()
        for index in range(15):
            self.assertEqual(self.feed(pose() + [index * 0.03, 0]), [])
        self.detector.reset()
        self.prepare()
        for index in range(20):
            points = pose()
            if index == 5:
                points[20, 1] += 0.65
            self.assertEqual(self.feed(points), [])

    def test_rotation_of_the_whole_hand_does_not_type(self):
        self.prepare()
        for angle in np.linspace(0, 0.5, 20):
            rotation = np.asarray([[np.cos(angle), -np.sin(angle)], [np.sin(angle), np.cos(angle)]])
            self.assertEqual(self.feed((pose() - [4, 3]) @ rotation.T + [4, 3]), [])

    def test_held_finger_does_not_repeat(self):
        self.prepare()
        clicks = []
        for _ in range(30):
            clicks.extend(self.feed(example(coupled=None).points[3]))
        self.assertEqual(clicks, [])

    def test_repeated_separate_presses_type_twice(self):
        self.prepare()
        clicks = []
        for _ in range(2):
            for points in example(coupled=None).points:
                clicks.extend(self.feed(points))
            for _ in range(4):
                clicks.extend(self.feed())
        self.assertEqual([c.key.name for c in clicks], ["a", "a"])

    def test_loss_or_time_gap_cancels_press(self):
        for gap in (False, True):
            self.detector.reset()
            self.prepare()
            self.feed(example(coupled=None).points[3])
            self.feed(example(coupled=None).points[3])
            if gap:
                self.time += 1
            else:
                self.time += 0.12
                self.detector.update(HandSnapshot(self.time, ()))
            for _ in range(5):
                self.assertEqual(self.feed(), [])

    def test_reach_to_another_row_selects_destination_instead_of_home(self):
        self.prepare()
        clicks = []
        # Pinky reaches from A (2.25, 2.5) to Q (2.25, 1.5), then lifts.
        for progress in (0, .35, .8, 1, 1, .8, .5, .5, .5):
            points = pose()
            points[20, 1] -= progress
            clicks.extend(self.feed(points))
        self.assertEqual([c.key.name for c in clicks], ["q"])

    def test_sideways_reach_and_return_does_not_type(self):
        self.prepare()
        for progress in (0, .35, .8, 1, 1, .8, .35, 0, 0):
            points = pose()
            points[20, 0] += progress
            self.assertEqual(self.feed(points), [])

    def test_tap_after_sideways_reposition_uses_the_new_key(self):
        self.prepare()
        for progress in (0, .35, .8, 1, 1, 1):
            points = pose()
            points[20, 0] += progress
            self.assertEqual(self.feed(points), [])
        clicks = []
        for points in example(coupled=None).points:
            points[20, 0] += 1
            clicks.extend(self.feed(points))
        self.assertEqual([c.key.name for c in clicks], ["s"])

    def test_overhead_depth_motion_works_with_nearly_stationary_xy(self):
        self.prepare()
        clicks = []
        for progress in (0, .35, .8, 1, 1, .8, .35, 0, 0):
            self.time += .12
            hand = snapshot(self.time).hands[0]
            points = list(hand.landmarks)
            p = points[20]
            points[20] = Landmark(p.x, p.y, .065 * progress)
            clicks.extend(self.detector.update(HandSnapshot(self.time, (TrackedHand(1, "Left", tuple(points)),))))
        self.assertEqual([c.key.name for c in clicks], ["a"])

    def test_depth_spike_and_rigid_hand_tilt_do_not_type(self):
        self.prepare()
        xyz = np.array([(p.x * 3, p.y, p.z * 3) for p in snapshot(0).hands[0].landmarks])
        for index, angle in enumerate(np.r_[np.linspace(0, .5, 15), np.linspace(.5, 0, 15)]):
            rotation = np.array([[1, 0, 0], [0, np.cos(angle), -np.sin(angle)],
                                 [0, np.sin(angle), np.cos(angle)]])
            points = (xyz - xyz[0]) @ rotation.T + xyz[0]
            if index == 5:
                points[20, 2] += .195
            self.time += .04
            hand = TrackedHand(1, "Left", tuple(Landmark(x / 3, y, z / 3) for x, y, z in points))
            self.assertEqual(self.detector.update(HandSnapshot(self.time, (hand,))), [])

    def test_quick_repeated_taps_at_phone_frame_rate(self):
        self.prepare()
        clicks = []
        for _ in range(3):
            for progress in (0, .4, .8, 1, .7, .2, 0, 0, 0):
                self.time += 1 / 30
                points = pose()
                points[20, 1] += .65 * progress
                clicks.extend(self.detector.update(snapshot(self.time, points)))
        self.assertEqual([c.key.name for c in clicks], ["a"] * 3)

    def test_staggered_fingers_can_overlap_on_the_same_hand(self):
        self.prepare()
        clicks = []
        wave = np.r_[np.linspace(0, 1, 6), np.linspace(1, 0, 6)]
        for frame in range(25):
            points = pose()
            for tip, start in ((20, 0), (16, 6)):
                i = frame - start
                if 0 <= i < len(wave):
                    points[tip, 1] += .65 * wave[i]
            self.time += 1 / 30
            clicks.extend(self.detector.update(snapshot(self.time, points)))
        self.assertEqual([c.key.name for c in clicks], ["a", "s"])

    def test_reaching_into_a_gap_does_not_fall_back_to_home_key(self):
        self.prepare()
        clicks = []
        for progress in (0, .35, .8, 1, 1, .8, .35, 0, 0):
            points = pose()
            points[20, 1] -= .5 * progress  # y=2.0 is the gap above A.
            clicks.extend(self.feed(points))
        self.assertEqual(clicks, [])

    def test_nearer_row_reach_can_settle_then_tap(self):
        home = pose()
        home[8, 0] = 5.5  # Inside F, also inside V on the row below.
        for _ in range(5):
            self.feed(home)
        for progress in (0, .35, .8, 1, 1, 1, 1, 1, 1):
            points = home.copy()
            points[8, 1] += progress
            self.assertEqual(self.feed(points), [])
        clicks = []
        for progress in (0, .35, .8, 1, .8, .35, 0, 0, 0):
            points = home.copy()
            points[8, 1] += 1 + .65 * progress
            clicks.extend(self.feed(points))
        self.assertEqual([c.key.name for c in clicks], ["v"])

    def test_partial_release_can_rearm_at_low_frame_rate(self):
        self.prepare()
        clicks = []
        for progress in (0, .35, .8, 1, 1, .8, .5, .5, .5, .5, .5):
            points = pose()
            points[20, 1] -= progress
            clicks.extend(self.feed(points))
        self.assertFalse(self.detector.states[1][4].releasing)
        self.assertEqual([c.key.name for c in clicks], ["q"])

    def jittered(self, rng, index_offset=0.0):
        # Hovering hands at phone frame rate: every fingertip wobbles by about
        # a tenth of a key from one frame to the next.
        self.time += 1 / 13
        points = pose()
        points[[4, 8, 12, 16, 20], 1] += rng.uniform(-0.1, 0.1, 5)
        points[8, 1] += index_offset
        return self.detector.update(snapshot(self.time, points))

    def test_hovering_finger_that_shifts_and_wobbles_does_not_type(self):
        # Screen recording: a still hovering hand typed J and L while only the
        # other hand moved. The fingertip had shifted a little over the same
        # key, and a one-frame wobble back looked like a press and release.
        for seed in range(8):
            rng = np.random.default_rng(seed)
            self.detector.reset()
            clicks = []
            for _ in range(20):
                clicks.extend(self.jittered(rng))
            for _ in range(50):
                clicks.extend(self.jittered(rng, index_offset=0.36))
            self.assertEqual(clicks, [], seed)

    def test_taps_still_type_through_tracking_wobble(self):
        for seed in range(8):
            rng = np.random.default_rng(seed)
            self.detector.reset()
            clicks = []
            for _ in range(20):
                clicks.extend(self.jittered(rng))
            for _ in range(4):
                for progress in (.5, 1, .5, 0):
                    clicks.extend(self.jittered(rng, index_offset=.65 * progress))
                for _ in range(8):
                    clicks.extend(self.jittered(rng))
            self.assertEqual([c.key.name for c in clicks], ["f"] * 4, seed)

    def test_whole_hand_move_then_immediate_tap_uses_new_position(self):
        self.prepare()
        self.assertEqual(self.feed(pose() + [2, 0]), [])
        clicks = []
        for points in example(coupled=None).points[1:]:
            clicks.extend(self.feed(points + [2, 0]))
        self.assertEqual([c.key.name for c in clicks], ["d"])


if __name__ == "__main__":
    unittest.main()
