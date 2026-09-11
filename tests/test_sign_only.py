import unittest

from test_traffic_behavior import Scene


class SignOnlyTests(unittest.TestCase):
    def test_branch_actions_are_refused_without_line_geometry(self):
        for label in ('left', 'right', 'cross', 'roadblock'):
            with self.subTest(label=label):
                s = Scene(follow_line=False)
                s.confirm(label)
                self.assertTrue(s.p.fault)
                self.assertEqual(s.command, (0, 0, 0))

    def test_round_and_u_turn_are_always_refused(self):
        for label in ('round', 'turn'):
            with self.subTest(label=label):
                s = Scene(follow_line=False)
                s.confirm(label)
                self.assertTrue(s.p.fault)
                self.assertEqual(s.command, (0, 0, 0))

    def test_zero_ceiling_cannot_consume_or_execute_sign(self):
        s = Scene(ceiling=0, follow_line=False)
        for label in ('left', 'red', 'green', 'red40'):
            s.confirm(label)
        self.assertEqual(s.p.state, 'driving')
        self.assertEqual(s.command, (0, 0, 0))


if __name__ == '__main__':
    unittest.main()
