"""Review-error interventions and denominators, independent of trained accuracy."""

import copy
import unittest

import numpy as np
from scipy import ndimage as ndi

from confirmseg.method import State
from confirmseg.review import corrupt_review, make_plan, statistics, validate_plan


class ReviewChecks(unittest.TestCase):
    def test_homogeneous_connected_bounded_subpatch(self):
        truth = np.zeros((64, 64), dtype=int)
        truth[:, 32:] = 1
        coordinates = np.argwhere(np.ones((6, 6), bool)) + (20, 29)
        action = corrupt_review(coordinates, truth, np.random.default_rng(7), 1,
                                "subpatch", .25, 5)
        wrong = np.asarray(action["labels"]) != truth[tuple(coordinates.T)]
        self.assertEqual(int(wrong.sum()), 5)
        self.assertEqual(len(np.unique(truth[tuple(coordinates[wrong].T)])), 1)
        mask = np.zeros_like(truth, bool)
        mask[tuple(coordinates[wrong].T)] = True
        self.assertEqual(ndi.label(mask, np.ones((3, 3)))[1], 1)
        self.assertFalse(wrong.all())
        condition = dict(wrong_mode="subpatch", wrong_subpatch_fraction=.25,
                         wrong_subpatch_max_pixels=5)
        validate_plan([action], truth, condition)
        invalid = copy.deepcopy(action)
        invalid["review"]["flipped_pixels"] += 1
        with self.assertRaisesRegex(ValueError, "metadata"):
            validate_plan([invalid], truth, condition)

    def test_shared_plan_geometry_rate_and_prefix(self):
        truth = np.indices((64, 64)).sum(0) % 2
        clean = make_plan(truth, "patient_task_0", 19, 12, 0)
        wrong = make_plan(truth, "patient_task_0", 19, 12, 1)
        prefix = make_plan(truth, "patient_task_0", 19, 3, 1)
        self.assertEqual(wrong[:3], prefix)
        self.assertEqual([a["coordinates"] for a in clean],
                         [a["coordinates"] for a in wrong])
        self.assertTrue(all(a["review"]["flipped_pixels"] == 0 for a in clean))
        self.assertTrue(all(a["review"]["flipped_pixels"] > 0 for a in wrong))
        # The chosen 8-connected subpatch can be limited by the same-class component.
        isolated = np.zeros((64, 64), int)
        isolated[20, 20] = 1
        action = corrupt_review(np.array([[20, 20]]), isolated,
                                np.random.default_rng(3), 1)
        self.assertEqual(action["review"]["flipped_pixels"], 1)

    def test_wrong_label_is_locked_until_explicit_user_revocation(self):
        truth = np.zeros((64, 64), int)
        coordinates = np.argwhere(np.ones((6, 6), bool)) + (20, 20)
        action = corrupt_review(coordinates, truth, np.random.default_rng(5), 1,
                                "subpatch", .25, 5)
        state = State.initial(np.full(truth.shape, .2, np.float32))
        self.assertEqual(state.act(action, (1, 1))[0], "confirm")
        wrong = np.asarray(action["review"]["flipped_coordinates"])
        np.testing.assert_array_equal(state.labels[tuple(wrong.T)], 1)
        correction = dict(type="correct", coordinates=wrong.tolist(), labels=[0]*len(wrong))
        self.assertEqual(state.act(correction, (1, 1))[0], "conflict")
        self.assertEqual(state.actions, 3)
        self.assertEqual(state.reviewed_area, 36)
        revoked = dict(type="revoke", coordinates=wrong.tolist())
        blocked = copy.deepcopy(state)
        self.assertEqual(blocked.act(revoked, (1, 1), disable_revocation=True)[0],
                         "revocation_disabled")
        np.testing.assert_array_equal(blocked.labels[tuple(wrong.T)], 1)
        self.assertEqual(state.act(revoked, (1, 1))[0], "revoke")
        self.assertEqual(state.act(correction, (1, 1))[0], "correct")
        self.assertEqual(state.actions, 5)
        self.assertEqual(state.channels((1, 1)).shape[0], 7)
        # Audit metadata never enters State.act's decisions or the seven model channels.
        without_metadata = copy.deepcopy(action)
        without_metadata.pop("review")
        a, b = (State.initial(np.full(truth.shape, .2, np.float32)) for _ in range(2))
        a.act(action, (1, 1))
        b.act(without_metadata, (1, 1))
        np.testing.assert_array_equal(a.channels((1, 1)), b.channels((1, 1)))

    def test_hand_computed_event_and_pixel_denominators(self):
        def row(index, verdict, wrong):
            return dict(verdict=verdict, review_event=dict(event_index=index,
                        review_pixels=36, wrong_pixels=wrong))

        episodes = [dict(snapshots=[row(0, "conflict", 5), row(0, "confirm", 5),
                                    row(1, "confirm", 0)])]
        result = statistics(episodes)
        self.assertEqual(result["review_attempts"], 3)
        self.assertEqual(result["unique_review_events"], 2)
        self.assertEqual(result["wrong_review_events"], 1)
        self.assertEqual(result["wrong_event_percent"], 50)
        self.assertEqual(result["accepted_review_pixels"], 72)
        self.assertEqual(result["accepted_wrong_review_pixels"], 5)
        self.assertAlmostEqual(result["accepted_wrong_pixel_percent"], 100*5/72)


if __name__ == "__main__":
    unittest.main()
