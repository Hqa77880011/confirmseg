"""Numerical/protocol checks, independent of medical data and model performance."""

import io
import copy
import unittest
from unittest.mock import patch

import numpy as np
import torch

from confirmseg.data import evaluation_tasks, validate_splits
from confirmseg.method import (
    Editor,
    InitialModel,
    State,
    band_map,
    footprint_target,
    head_context,
    loss_terms,
    infer,
    physical_dilation,
)
from confirmseg.metrics import (
    assd,
    budget_integral,
    counts,
    episode_statistics,
    overlap,
    paired_intervals,
    rework_update,
    seed_statistics,
    transition,
)


def metric_episode(patient="p", identifier="task", tp=9, fn=2, rework=0):
    return {
        "id": identifier,
        "patient_id": patient,
        "dataset": "check",
        "reader": 0,
        "budget": 12,
        "snapshots": [
            {
                "tp": tp,
                "fp": 0,
                "fn": fn,
                "action": 12,
                "rework": rework,
                "reviewed_area_fraction": 0.01,
            }
        ],
    }


class ProtocolChecks(unittest.TestCase):
    def test_console_success_does_not_return_a_model(self):
        from confirmseg.cli import main

        with patch("confirmseg.cli.train", return_value=object()):
            result = main(
                [
                    "train",
                    "--stage",
                    "initial",
                    "--manifest",
                    "unused",
                    "--output",
                    "unused",
                ]
            )
        self.assertIsNone(result)

    def test_conflict_revocation_and_charging(self):
        state = State.initial(np.full((32, 32), 0.2, np.float32))
        review = {"type": "confirm", "coordinates": [[5, 5]], "labels": [1]}
        self.assertEqual(state.act(review, (1, 1))[0], "confirm")
        self.assertEqual(state.probability[5, 5], 1)
        correction = dict(review, type="correct", labels=[0])
        channels = state.channels((1, 1)).copy()
        self.assertEqual(state.act(correction, (1, 1))[0], "conflict")
        np.testing.assert_array_equal(channels, state.channels((1, 1)))
        self.assertEqual(state.actions, 3)
        state.act({"type": "revoke", "coordinates": [[5, 5]]}, (1, 1))
        self.assertEqual(state.labels[5, 5], -1)
        self.assertEqual(state.revoked[5, 5], 1)
        self.assertEqual(state.probability[5, 5], 1)
        self.assertEqual(state.act(correction, (1, 1))[0], "correct")
        self.assertEqual(state.actions, 5)
        self.assertEqual(state.negative[5, 5], 1)
        self.assertEqual(state.reviewed_area, 1)

    def test_physical_geometry_and_empty_target(self):
        point = np.zeros((7, 7), bool)
        point[3, 3] = True
        dilation = physical_dilation(point, 1, (2, 1))
        self.assertEqual(int(dilation.sum()), 3)
        self.assertFalse(dilation[2, 3])
        self.assertTrue(dilation[3, 2])
        self.assertFalse(band_map(np.zeros_like(point), (2, 1)).any())
        truth = point.astype(float)
        target = footprint_target(
            np.zeros_like(truth), truth, np.zeros_like(point), point, (2, 1), 1
        )
        np.testing.assert_array_equal(target, dilation)
        self.assertFalse(
            footprint_target(
                np.zeros_like(truth), truth, point, np.zeros_like(point), (2, 1), 1
            ).any()
        )

    def test_patient_split_and_stratified_tasks(self):
        with self.assertRaisesRegex(ValueError, "crosses splits"):
            validate_splits(
                [
                    dict(id="a", patient_id="p", dataset="ACDC", split="train"),
                    dict(id="b", patient_id="p", dataset="ACDC", split="test"),
                ]
            )
        rows = [
            dict(
                id=f"{phase}_{structure}_{z}",
                patient_id="p",
                dataset="ACDC",
                phase=phase,
                structure=structure,
            )
            for phase in ("ED", "ES")
            for structure in (1, 2, 3)
            for z in (0, 1)
        ]
        tasks = evaluation_tasks(rows)
        self.assertEqual(len(tasks), 5)
        self.assertEqual(len({(row["phase"], row["structure"]) for row, _ in tasks}), 5)
        self.assertEqual(tasks, evaluation_tasks(rows))


class NumericalChecks(unittest.TestCase):
    def test_overlap_and_surface_distance(self):
        values = counts(np.array([1, 1, 1, 0]), np.array([1, 1, 0, 1]))
        self.assertEqual(values, dict(tp=2, fp=1, fn=1))
        dice, iou = overlap(values)
        self.assertAlmostEqual(dice, 2 / 3)
        self.assertAlmostEqual(iou, 0.5)
        self.assertAlmostEqual(dice, 2 * iou / (1 + iou))
        a, b = np.zeros((4, 4)), np.zeros((4, 4))
        a[1, 1], b[2, 1] = 1, 1
        self.assertEqual(assd(a, b, np.array([2, 1])), (2.0, False))
        self.assertEqual(assd(a * 0, b * 0, np.array([2, 1])), (0.0, False))
        distance, failed = assd(a * 0, b, np.array([2, 1]))
        self.assertTrue(failed)
        self.assertAlmostEqual(distance, np.sqrt(80))

    def test_hand_computed_five_term_loss(self):
        old = torch.tensor([[[[1.0, 0.2], [0.4, 0.8]]]])
        truth = torch.tensor([[[[1.0, 0], [1, 1]]]])
        q = torch.tensor([[[[0.1, 0.6], [0.8, 0.4]]]])
        gate = torch.tensor([[[[0.0, 0.5], [0.25, 0.5]]]])
        edited = torch.tensor([[[[1.0, 0.4], [0.5, 0.6]]]])
        target = torch.tensor([[[[0.0, 1], [1, 0]]]])
        state = torch.zeros(1, 7, 2, 2)
        state[:, :1], state[:, 3, 0, 0], state[:, 6] = old, 1, 1
        weights = dict(gate=1, area=0.01, regression=0.5, boundary=0.1)
        actual = loss_terms(
            edited, q, gate, state, truth, target, (2, 1), weights, "confirmseg"
        )
        bce = (-np.log(1 - 1e-6) - np.log(0.6) - np.log(0.5) - np.log(0.6)) / 4
        dice_loss = 1 - (4.2 + 1e-6) / (5.5 + 1e-6)
        gate_loss = (-np.log(0.5) - np.log(0.25) - np.log(0.5)) / 3
        expected = (
            bce + dice_loss + gate_loss + 0.01 * 1.25 / 3 + 0.5 * 0.2 + 0.1 * 0.2875
        )
        self.assertAlmostEqual(float(actual), expected, places=6)

    def test_repair_changed_and_regression_denominators(self):
        truth, old = np.zeros((3, 3)), np.zeros((3, 3))
        truth[1, 1] = 1
        updated = truth.copy()
        updated[0, 0] = 1
        support = truth.astype(bool)
        values = transition(old, updated, truth, np.full((3, 3), -1), support, (1, 1))
        self.assertEqual(
            (values["corrected_target_pixels"], values["target_error_pixels"]), (1, 1)
        )
        self.assertEqual(
            (values["regression_numerator"], values["prior_correct_unconfirmed"]),
            (1, 8),
        )
        self.assertEqual((values["changed_pixels"], values["editable_pixels"]), (2, 9))
        self.assertTrue(
            all(pair == [0, 0] for pair in values["band_regression"].values())
        )

    def test_distance_bands_exclude_approvals(self):
        truth = np.zeros((5, 9))
        old = truth.copy()
        updated = truth.copy()
        updated[2, 1], updated[2, 4], updated[2, 7] = 1, 1, 1
        labels = np.full(truth.shape, -1)
        labels[:, 0] = 0
        values = transition(
            old, updated, truth, labels, np.ones_like(truth, bool), (1, 1)
        )
        self.assertEqual(
            values["band_regression"],
            {"0_3mm": [1, 10], "3_6mm": [1, 15], "over_6mm": [1, 15]},
        )

    def test_first_hit_cost_uses_hit_area(self):
        episode = metric_episode()
        episode["snapshots"] = [
            dict(tp=0, fp=0, fn=1, action=1, reviewed_area_fraction=0, rework=0),
            dict(tp=1, fp=0, fn=0, action=2, reviewed_area_fraction=0.01, rework=0),
            dict(tp=0, fp=0, fn=1, action=12, reviewed_area_fraction=0.1, rework=2),
        ]
        values = episode_statistics(episode)
        self.assertEqual(values["actions90"], 2)
        self.assertAlmostEqual(values["cost90_alpha_10"], 2.1)
        episode["snapshots"] = episode["snapshots"][::2]
        self.assertEqual(episode_statistics(episode)["cost90_alpha_50"], 18)

    def test_rework_compares_each_touched_component(self):
        truth, old = np.zeros((3, 7)), np.zeros((3, 7))
        truth[1, 1], truth[1, 5] = 1, 1
        a, b = truth.copy().astype(bool), truth.copy().astype(bool)
        a[1, 5], b[1, 1] = False, False
        self.assertEqual(
            rework_update([a, b], old, truth, truth, truth.astype(bool)), 1
        )

    def test_paired_cluster_intervals_and_seed_sd(self):
        runs = []
        for seed in (17, 29, 43):
            for method, tp, fn, rework in (("confirmseg", 9, 2, 1), ("dmp", 4, 2, 2)):
                runs.append(
                    dict(
                        method=method,
                        seed=seed,
                        episodes=[
                            metric_episode(patient, patient, tp, fn, rework)
                            for patient in ("p1", "p2")
                        ],
                    )
                )
        result = paired_intervals(runs, ("confirmseg", "dmp"), 100)["check"]
        np.testing.assert_allclose(
            result["dice_pp"]["patient_interval95"], [10, 10], atol=1e-12
        )
        self.assertAlmostEqual(result["dice_pp"]["difference"], 10)
        self.assertAlmostEqual(result["dice_pp"]["seed_sd"], 0)
        self.assertEqual(result["repeated_corrections"]["difference"], -1)
        self.assertEqual(
            seed_statistics([dict(x=1), dict(x=3), dict(x=5)])["x"]["sd"], 2
        )
        self.assertAlmostEqual(
            budget_integral(
                [
                    dict(budget=1, dice=0.2),
                    dict(budget=3, dice=0.6),
                    dict(budget=5, dice=1),
                ],
                5,
            ),
            0.6,
        )


class ModelChecks(unittest.TestCase):
    def test_training_and_inference_merge_agree_for_every_method(self):
        from confirmseg.method import METHODS

        torch.set_num_threads(2)
        torch.manual_seed(17)
        initial, editor = InitialModel().eval(), Editor().eval()
        image = np.random.default_rng(17).normal(size=(32, 32)).astype(np.float32)
        state = State.initial(np.full(image.shape, 0.3, np.float32))
        state.labels[4:6, 4:6] = 1
        state.probability[4:6, 4:6] = 1
        state.positive[16, 16] = 1
        support = state.positive > 0
        spacing = np.array([1.0, 2.0])
        with torch.inference_mode():
            features = initial.encoder(torch.from_numpy(image)[None, None])
            channels = torch.from_numpy(state.channels(spacing))[None]
            for method in METHODS:
                expected = editor(
                    features, channels, method, spacing=spacing, support=support
                )[0][0, 0].numpy()
                edited_state = copy.deepcopy(state)
                infer(
                    initial,
                    editor,
                    image,
                    edited_state,
                    spacing,
                    method,
                    "cpu",
                    features=features,
                    prompt_support=support,
                )
                np.testing.assert_allclose(
                    edited_state.probability,
                    expected,
                    atol=1e-6,
                    rtol=0,
                    err_msg=method,
                )

    def test_gradients_projection_bound_and_checkpoint_roundtrip(self):
        torch.set_num_threads(2)
        torch.manual_seed(17)
        initial, editor = InitialModel().eval(), Editor()
        initial.requires_grad_(False)
        image = torch.rand(1, 1, 32, 32)
        with torch.no_grad():
            features = initial.encoder(image)
        state = torch.zeros(1, 7, 32, 32)
        state[:, 0] = 0.3
        state[:, 3, 10:12, 10:12] = 1
        state[:, 4, :2, :2] = 1
        truth = torch.zeros(1, 1, 32, 32)
        truth[:, :, 8:20, 8:20] = 1
        before = {
            name: parameter.detach().clone()
            for name, parameter in editor.named_parameters()
        }
        optimizer = torch.optim.AdamW(editor.parameters(), lr=1e-3)
        edited, candidate, gate = editor(features, state)
        target = torch.zeros_like(truth)
        target[:, :, 8:20, 8:20] = 1
        loss = loss_terms(
            edited,
            candidate,
            gate,
            state,
            truth,
            target,
            (1, 1),
            dict(gate=1, area=0.01, regression=0.5, boundary=0.1),
            "confirmseg",
        )
        self.assertTrue(torch.isfinite(loss))
        loss.backward()
        for name, module in (("candidate", editor.candidate), ("gate", editor.gate)):
            gradients = [p.grad for p in module.parameters()]
            self.assertTrue(
                all(g is not None and torch.isfinite(g).all() for g in gradients), name
            )
            self.assertGreater(sum(float(g.abs().sum()) for g in gradients), 0)
        self.assertTrue(all(p.grad is None for p in initial.parameters()))
        optimizer.step()
        for name in ("candidate.output.weight", "gate.4.weight"):
            self.assertFalse(
                torch.equal(before[name], dict(editor.named_parameters())[name])
            )
        approved = state[:, 3:5].sum(1, keepdim=True).bool()
        self.assertTrue(torch.equal(edited[approved], state[:, 3:4][approved]))
        self.assertEqual(float(gate[approved].sum()), 0)
        self.assertLessEqual(
            float((edited - state[:, :1]).abs()[~approved].sum()),
            float(gate.sum()) + 1e-5,
        )
        editor.eval()
        with torch.inference_mode():
            expected = editor(features, state)[0]
        buffer = io.BytesIO()
        torch.save(editor.state_dict(), buffer)
        buffer.seek(0)
        restored = Editor().eval()
        restored.load_state_dict(torch.load(buffer, weights_only=True))
        with torch.inference_mode():
            actual = restored(features, state)[0]
        torch.testing.assert_close(actual, expected, rtol=0, atol=1e-6)
        for method, head in (
            ("no_approval", "candidate"),
            ("no_approval", "gate"),
            ("gate_only", "candidate"),
            ("candidate_only", "gate"),
        ):
            context = head_context(state, method, head)
            self.assertEqual(float(context[:, 3:].sum()), 0)
            torch.testing.assert_close(context[:, :3], state[:, :3])


if __name__ == "__main__":
    unittest.main()
