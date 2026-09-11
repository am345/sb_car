import unittest

from core.traffic_behavior import SignEvents


def frame(sequence, labels=(), age=0):
    return {'state': 'ready', 'age_sec': age, 'sequence': sequence,
            'detections': [{'label': label, 'confidence': .9}
                           for label in labels]}


class CooldownTests(unittest.TestCase):
    def test_execution_does_not_rearm_visible_sign(self):
        events = SignEvents()
        for sequence in range(1, 4):
            ready = events.update(frame(sequence, ['left']), sequence * .1)
        self.assertEqual(ready, ['left'])
        events.begin('left', .3)
        for sequence in range(4, 100):
            self.assertEqual(events.update(frame(sequence, ['left']), sequence * .1), [])
        self.assertEqual(events.status(10)['left']['phase'], 'executing')

    def test_cooldown_starts_when_action_finishes(self):
        events = SignEvents()
        for sequence in range(1, 4):
            events.update(frame(sequence, ['cross']), sequence * .1)
        events.begin('cross', .3)
        self.assertEqual(events.status(20)['cross']['phase'], 'executing')
        events.complete('cross', 20)
        self.assertAlmostEqual(events.status(20)['cross']['remaining_sec'], 5)

    def test_rearm_requires_post_cooldown_absence(self):
        events = SignEvents()
        events.begin('right', 0)
        events.complete('right', 0)
        events.update(frame(1), 5.1)
        events.update(frame(2), 5.9)
        events.update(frame(3), 6.7)
        self.assertNotIn('right', events.locks)
        for sequence in (4, 5):
            self.assertEqual(events.update(frame(sequence, ['right']), 7 + sequence / 10), [])
        self.assertEqual(events.update(frame(6, ['right']), 7.6), ['right'])

    def test_stale_data_is_not_clearance_evidence(self):
        events = SignEvents()
        events.begin('red', 0)
        events.complete('red', 0)
        events.update(frame(1, age=2), 6)
        self.assertIn('red', events.locks)


if __name__ == '__main__':
    unittest.main()
