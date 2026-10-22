"""Regression and rejection tests for the released core method."""

import importlib.util
import unittest
from copy import deepcopy
from fractions import Fraction

from overcooked_cooperation import (
    certify_cooperation,
    check_bundle,
    finite_bound,
    make_model,
    solve_model,
)
from overcooked_cooperation.api import check_model
from overcooked_cooperation.execution import (
    finite_output,
    qualify_layout,
    run_schedule,
    verify_schedule,
)
from overcooked_cooperation.io import fingerprint

HAS_SOLVER = importlib.util.find_spec("scipy") is not None
HAS_NATIVE = importlib.util.find_spec("numpy") is not None


class ModelTests(unittest.TestCase):
    def test_recorded_original_dimensions(self):
        expected = {
            "ring": (257, 1464),
            "asymmetric": (293, 1313),
            "circuit": (446, 2592),
        }
        for layout, dimensions in expected.items():
            with self.subTest(layout=layout):
                model = make_model(layout)
                self.assertEqual(
                    (len(model["row_labels"]), model["variable_count"]), dimensions
                )
                self.assertTrue(check_model(model)["geometry_and_all_columns_checked"])

    def test_corrupted_column_is_rejected(self):
        model = make_model("ring")
        model["A_eq_sparse"][0][0][1] += 1
        with self.assertRaises(ValueError):
            check_model(model)

    def test_embedded_start_is_checked(self):
        model = make_model("ring")
        model["start_positions"].reverse()
        with self.assertRaises(ValueError):
            check_model(model)

    def test_component_eligibility(self):
        self.assertTrue(qualify_layout("asymmetric")["qualified"])
        self.assertFalse(qualify_layout("ring")["qualified"])
        with self.assertRaises(ValueError):
            run_schedule("ring", scheduling="asynchronous")


@unittest.skipUnless(HAS_SOLVER, "Install the solver extra")
class ExactBoundTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.ring = solve_model(make_model("ring"))

    def test_exact_ring_bound(self):
        result = check_bundle(self.ring)
        self.assertTrue(result["matching_exact_LP_optimum"])
        self.assertEqual(result["native_rate_upper"], "1/42")

    def test_empty_start_finite_bound(self):
        self.assertEqual(finite_bound(self.ring, 0)["upper_soups"], 0)
        self.assertEqual(finite_bound(self.ring, 200)["upper_soups"], 4)
        self.assertEqual(finite_bound(self.ring, 400)["upper_soups"], 9)
        with self.assertRaises(ValueError):
            finite_bound(self.ring, -1)

    def test_exact_checker_rejects_tiny_invalid_dual(self):
        bundle = deepcopy(self.ring)
        row = bundle["model"]["row_labels"].index("accepted_full_serve")
        bundle["certificate"]["dual_eq"][row] = str(
            Fraction(bundle["certificate"]["dual_eq"][row]) + Fraction(1, 10**30)
        )
        self.assertFalse(check_bundle(bundle)["dual_valid_exact"])

    def test_model_identity_is_checked(self):
        bundle = deepcopy(self.ring)
        bundle["model"]["cook_time"] = 19
        bundle["certificate"]["model_sha256"] = fingerprint(bundle["model"])
        with self.assertRaises(ValueError):
            check_bundle(bundle)

    def test_pot_state_bound(self):
        bundle = solve_model(make_model("asymmetric", "two_active", pot_states=True))
        self.assertEqual(bundle["result"]["native_rate_upper"], "2/23")
        self.assertTrue(bundle["result"]["matching_exact_LP_optimum"])
        with self.assertRaises(ValueError):
            finite_bound(bundle, 200)


@unittest.skipUnless(HAS_NATIVE, "Install the native extra")
class ScheduleTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.ring = run_schedule("ring")
        cls.asymmetric = run_schedule("asymmetric")

    def test_joint_word_and_observed_counts(self):
        self.assertEqual(self.ring["rate"], "3/97")
        self.assertEqual(self.ring["observed_steps"], 965)
        self.assertTrue(verify_schedule(self.ring)["native_replay_verified"])
        self.assertEqual(
            finite_output(self.ring, 200),
            {"horizon": 200, "soups": 5, "type": "observed"},
        )
        self.assertEqual(finite_output(self.ring, 400)["soups"], 11)

    def test_product_and_extrapolated_counts(self):
        schedule = self.asymmetric
        self.assertEqual(schedule["rate"], "19/450")
        self.assertEqual(schedule["observed_steps"], 152)
        self.assertEqual(schedule["joint_period"], 450)
        self.assertEqual(schedule["joint_period_type"], "constructed_from_local_words")
        self.assertTrue(verify_schedule(schedule)["native_replay_verified"])
        self.assertEqual(
            finite_output(schedule, 200),
            {"horizon": 200, "soups": 8, "type": "extrapolated"},
        )
        self.assertEqual(finite_output(schedule, 400)["soups"], 16)

    def test_edited_rate_is_rejected(self):
        schedule = deepcopy(self.asymmetric)
        schedule["rate"] = "1"
        with self.assertRaises(ValueError):
            verify_schedule(schedule)

    @unittest.skipUnless(HAS_SOLVER, "Install the solver extra")
    def test_complete_ring_certificate(self):
        solo0 = solve_model(make_model("ring", "actor0_only"))
        solo1 = solve_model(make_model("ring", "actor1_only"))
        result = certify_cooperation(solo0, solo1, self.ring)
        self.assertEqual(result["resource_gain_guarantee"], "29/97")
        self.assertTrue(result["at_least_ten_percent"])
        self.assertEqual([row["additional_soups"] for row in result["finite"]], [1, 2])


if __name__ == "__main__":
    unittest.main()
