"""The marketplace filter speaks different units than the marketplace replies.

A template encodes the floors so a future run inherits them without anyone
remembering why they exist. That only helps if the numbers mean what they look
like — and vast's `gpu_ram` filter takes decimal GB while its offers report MB,
a 7% disagreement. Filtering at the 27B's measured 46GB requirement admits a
46,068MB L40S, which is 44.99 GiB: the one card three rentals proved cannot
hold this model. The floor is therefore derived from feasible(), not typed."""
from launch.matrix import MODELS, TIERS, feasible
from launch.templates import vram_floor_decimal_gb


def test_the_floor_excludes_a_card_the_model_does_not_fit():
    """46068MB is what every Vast host reports for an L40S."""
    floor = vram_floor_decimal_gb(MODELS["headline"])
    assert 46068 / 1000 < floor, "an L40S must not pass the 27B's filter"
    assert feasible(MODELS["headline"], TIERS["L40S"]) is False


def test_the_floor_admits_every_card_the_model_does_fit():
    """Raising a floor is how a tier silently stops being rentable."""
    floor = vram_floor_decimal_gb(MODELS["headline"])
    for reported_mb in (81920, 97887):  # A100 80GB, RTX PRO 6000
        assert reported_mb / 1000 >= floor


def test_the_anchor_floor_admits_the_smallest_card_it_ran_on():
    """The 8B measured on a 32GB RTX 5090; a floor that excluded it would drop
    the cheapest tier in the matrix."""
    floor = vram_floor_decimal_gb(MODELS["anchor"])
    assert 32768 / 1000 >= floor


def test_the_floor_is_not_the_bare_requirement():
    """The naive value. vLLM reserves 10% and the units differ, so a filter set
    to the requirement itself rents cards that cannot serve the model."""
    assert vram_floor_decimal_gb(MODELS["headline"]) > MODELS["headline"].required_vram_gb


def test_the_floor_agrees_with_feasible_at_the_boundary():
    """One rule, two expressions: whatever the filter admits, feasible() must
    also accept, or the matrix and the template disagree about what is rentable."""
    floor = vram_floor_decimal_gb(MODELS["headline"])
    reported_gib = floor * 1000 / 1024
    assert MODELS["headline"].required_vram_gb <= reported_gib * 0.9
