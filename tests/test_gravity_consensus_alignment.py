from __future__ import annotations

import numpy as np

from atlas3r.mapping import _gravity_consensus_decision


def test_gravity_consensus_legacy_precedence():
    decision = _gravity_consensus_decision(
        legacy_reliable=True,
        consensus_enabled=True,
        consensus_normal=[0.0, 0.0, 1.0],
        camera_up_prior=[0.0, 0.0, 1.0],
        np=np,
    )

    assert decision["path"] == "legacy"
    assert decision["align_consensus"] is False
    assert decision["reason"] == "legacy_inlier_precedence"


def test_gravity_consensus_aligns_when_plane_agrees_with_prior():
    decision = _gravity_consensus_decision(
        legacy_reliable=False,
        consensus_enabled=True,
        consensus_normal=[0.0, 0.0, -1.0],
        camera_up_prior=[0.0, 0.08715574, 0.9961947],
        np=np,
    )

    assert decision["path"] == "consensus"
    assert decision["align_consensus"] is True
    assert decision["agreement_angle_deg"] < 6.0


def test_gravity_consensus_refuses_when_plane_disagrees_with_prior():
    decision = _gravity_consensus_decision(
        legacy_reliable=False,
        consensus_enabled=True,
        consensus_normal=[0.0, 0.5, 0.8660254],
        camera_up_prior=[0.0, 0.0, 1.0],
        np=np,
    )

    assert decision["path"] == "none"
    assert decision["align_consensus"] is False
    assert decision["reason"] == "plane_normal_disagrees_with_camera_up_prior"
    assert decision["agreement_angle_deg"] > 10.0


def test_gravity_consensus_can_be_disabled():
    decision = _gravity_consensus_decision(
        legacy_reliable=False,
        consensus_enabled=False,
        consensus_normal=[0.0, 0.0, 1.0],
        camera_up_prior=[0.0, 0.0, 1.0],
        np=np,
    )

    assert decision["path"] == "none"
    assert decision["align_consensus"] is False
    assert decision["reason"] == "gravity_consensus_alignment_disabled"
