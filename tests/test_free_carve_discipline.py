import pytest

from atlas3r.mapping import (
    FREE_CARVE_BAND_BOTTOM,
    FREE_CARVE_BAND_UPPER,
    _free_carve_discipline_decision,
)


def test_free_carve_off_preserves_unconditional_no_margin_carve():
    decision = _free_carve_discipline_decision(
        "off",
        terminal_verified=False,
        band_position=FREE_CARVE_BAND_BOTTOM,
        base_margin_m=0.1,
    )
    assert decision == {"create_free_evidence": True, "truncation_margin_m": 0.0}


def test_verified_only_rejects_unverified_free_evidence():
    assert not _free_carve_discipline_decision(
        "verified_only",
        terminal_verified=False,
        band_position=FREE_CARVE_BAND_UPPER,
        base_margin_m=0.1,
    )["create_free_evidence"]
    assert _free_carve_discipline_decision(
        "verified_only",
        terminal_verified=True,
        band_position=FREE_CARVE_BAND_UPPER,
        base_margin_m=0.1,
    )["create_free_evidence"]


def test_truncated_uses_base_margin_for_verified_and_3x_for_unverified():
    verified = _free_carve_discipline_decision(
        "truncated",
        terminal_verified=True,
        band_position=FREE_CARVE_BAND_BOTTOM,
        base_margin_m=0.1,
    )
    unverified = _free_carve_discipline_decision(
        "truncated",
        terminal_verified=False,
        band_position=FREE_CARVE_BAND_BOTTOM,
        base_margin_m=0.1,
    )
    assert verified == {"create_free_evidence": True, "truncation_margin_m": 0.1}
    assert unverified["create_free_evidence"]
    assert unverified["truncation_margin_m"] == pytest.approx(0.3)


def test_both_is_verified_only_in_bottom_half():
    assert not _free_carve_discipline_decision(
        "both",
        terminal_verified=False,
        band_position=FREE_CARVE_BAND_BOTTOM,
        base_margin_m=0.1,
    )["create_free_evidence"]
    assert _free_carve_discipline_decision(
        "both",
        terminal_verified=True,
        band_position=FREE_CARVE_BAND_BOTTOM,
        base_margin_m=0.1,
    ) == {"create_free_evidence": True, "truncation_margin_m": 0.0}


def test_both_is_truncated_outside_bottom_half():
    decision = _free_carve_discipline_decision(
        "both",
        terminal_verified=False,
        band_position=FREE_CARVE_BAND_UPPER,
        base_margin_m=0.1,
    )
    assert decision["create_free_evidence"]
    assert decision["truncation_margin_m"] == pytest.approx(0.3)
