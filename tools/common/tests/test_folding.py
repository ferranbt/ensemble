"""Tests for the shared predictor contract.

What matters here is substitutability: the same reading code has to work on any
predictor's result. So the tests feed each predictor's *native* score keys and
scales through the normalisers and assert one shape comes out, using the real
key names from Boltz's confidence json, Protenix's `summary_confidence` and
ESMFold2's result object.
"""

from __future__ import annotations

import pytest

from tools.common import chains as chain_spec
from tools.common import folding

TARGET = "MKTAYIAKQRQISFVK"
BINDER = "GHNLYQWRSTVEDCAP"


def complex_chains(**msa) -> dict[str, chain_spec.Chain]:
    return chain_spec.parse(
        {
            "A": {"sequence": TARGET, "msa": msa.get("a", "")},
            "B": {"sequence": BINDER, "msa": msa.get("b", "")},
        }
    )


class TestFraction:
    @pytest.mark.parametrize(
        "value,expected", [(0.95, 0.95), (95.0, 0.95), (1.0, 1.0), (0.0, 0.0)]
    )
    def test_normalises_either_scale(self, value, expected):
        """Protenix reports pLDDT out of 100, Boltz out of 1."""
        assert folding.fraction(value) == pytest.approx(expected)

    @pytest.mark.parametrize("value", [None, "", "abc", {}])
    def test_unusable_is_none_not_zero(self, value):
        """Zero confidence and absent confidence must not look alike."""
        assert folding.fraction(value) is None


class TestPairScores:
    def test_boltz_nested_index_strings(self):
        assert folding.pair_scores({"0": {"1": 0.9}}, ["A", "B"]) == {"A": {"B": 0.9}}

    def test_protenix_square_matrix(self):
        result = folding.pair_scores([[1.0, 0.91], [0.91, 1.0]], ["A", "B"])
        assert result == {
            "A": {"A": 1.0, "B": 0.91},
            "B": {"A": 0.91, "B": 1.0},
        }

    def test_both_forms_agree_on_the_pair(self):
        """The point of normalising: same number, whoever reported it."""
        nested = folding.pair_scores({"0": {"1": 0.91}}, ["A", "B"])
        matrix = folding.pair_scores([[1.0, 0.91], [0.91, 1.0]], ["A", "B"])
        assert nested["A"]["B"] == matrix["A"]["B"]

    def test_index_beyond_the_chain_list_is_dropped(self):
        """A ligand chain can appear in the matrix but not in the sequences."""
        assert folding.pair_scores({"0": {"5": 0.9}}, ["A", "B"]) == {}

    def test_empty(self):
        assert folding.pair_scores(None, ["A"]) == {}


class TestScores:
    def test_boltz_keys(self):
        raw = {
            "confidence_score": 0.87,
            "ptm": 0.91,
            "iptm": 0.96,
            "complex_plddt": 0.967,
            "pair_chains_iptm": {"0": {"1": 0.96}},
        }
        result = folding.scores(raw, ["A", "B"])
        assert result["confidence_score"] == 0.87
        assert result["complex_plddt"] == pytest.approx(0.967)
        assert result["chain_pair_iptm"] == {"A": {"B": 0.96}}

    def test_protenix_keys_map_to_the_same_names(self):
        raw = {
            "ranking_score": 0.89,
            "ptm": 0.88,
            "iptm": 0.91,
            "plddt": 67.5,
            "chain_pair_iptm": [[0.95, 0.91], [0.91, 0.93]],
        }
        result = folding.scores(raw, ["A", "B"])
        assert result["confidence_score"] == 0.89
        # Rescaled from Protenix's 0-100, so it is comparable with Boltz's.
        assert result["complex_plddt"] == pytest.approx(0.675)
        assert result["chain_pair_iptm"]["A"]["B"] == 0.91

    def test_esmfold2_keys(self):
        result = folding.scores({"ptm": 0.96, "iptm": 0.955, "plddt": 0.949}, ["A", "B"])
        assert result["complex_plddt"] == pytest.approx(0.949)
        assert result["iptm"] == pytest.approx(0.955)

    def test_all_three_produce_the_same_key_set(self):
        """A consumer reads one set of names regardless of the producer."""
        boltz = folding.scores(
            {"confidence_score": 0.8, "ptm": 0.9, "iptm": 0.9, "complex_plddt": 0.9},
            ["A", "B"],
        )
        protenix = folding.scores(
            {"ranking_score": 0.8, "ptm": 0.9, "iptm": 0.9, "plddt": 90.0}, ["A", "B"]
        )
        assert set(boltz) == set(protenix)

    def test_missing_scores_are_absent_not_zero(self):
        assert folding.scores({"iptm": 0.9}, ["A", "B"]) == {"iptm": 0.9}

    def test_prefers_the_standard_name_over_the_alias(self):
        raw = {"complex_plddt": 0.9, "plddt": 0.1}
        assert folding.scores(raw, ["A"])["complex_plddt"] == 0.9


class TestInterface:
    def test_finds_the_pair_either_way_round(self):
        model = {"chain_pair_iptm": {"A": {"B": 0.93}}}
        assert folding.interface(model, "A", "B") == 0.93
        assert folding.interface(model, "B", "A") == 0.93

    def test_absent_is_none(self):
        assert folding.interface({}, "A", "B") is None

    def test_reads_what_scores_produced(self):
        """The regression that motivated this: Boltz keys pairs by index, so
        looking them up by chain id returned None on every run."""
        model = folding.scores({"pair_chains_iptm": {"0": {"1": 0.96}}}, ["A", "B"])
        assert folding.interface(model, "A", "B") == 0.96


class TestSingleSequenceOnly:
    def test_accepts_chains_with_no_alignment(self):
        folding.single_sequence_only(complex_chains(), "ESMFold2")

    def test_refuses_rather_than_ignores(self):
        """Silently dropping it would report a weaker prediction as a stronger one."""
        with pytest.raises(ValueError, match="cannot use an alignment"):
            folding.single_sequence_only(
                complex_chains(a="s3://bucket/a.a3m"), "ESMFold2"
            )

    def test_names_the_offending_chains(self):
        with pytest.raises(ValueError, match="A, B"):
            folding.single_sequence_only(
                complex_chains(a="colabfold", b="colabfold"), "ESMFold2"
            )


class TestOneServer:
    def test_none_named(self):
        assert folding.one_server(complex_chains(), "Boltz") == ""

    def test_one_named(self):
        chains = complex_chains(a="https://api.colabfold.com")
        assert folding.one_server(chains, "Boltz") == "https://api.colabfold.com"

    def test_precomputed_alignments_are_not_servers(self):
        chains = complex_chains(a="s3://bucket/a.a3m", b="s3://bucket/b.a3m")
        assert folding.one_server(chains, "Boltz") == ""

    def test_two_servers_refused(self):
        chains = complex_chains(
            a="https://api.colabfold.com", b="https://other.example.com"
        )
        with pytest.raises(ValueError, match="one alignment server per job"):
            folding.one_server(chains, "Boltz")


class TestSharedAlignmentWarning:
    def test_warns_when_different_sequences_share_one(self, capsys):
        folding.shared_alignment_warning(
            complex_chains(a="s3://bucket/a.a3m", b="s3://bucket/a.a3m")
        )
        assert "share the alignment" in capsys.readouterr().out

    def test_silent_when_sequences_are_identical(self, capsys):
        """A homodimer sharing one alignment is correct, not a mistake."""
        chains = chain_spec.parse(
            {
                "A": {"sequence": TARGET, "msa": "s3://bucket/a.a3m"},
                "B": {"sequence": TARGET, "msa": "s3://bucket/a.a3m"},
            }
        )
        folding.shared_alignment_warning(chains)
        assert capsys.readouterr().out == ""

    def test_silent_when_each_has_its_own(self, capsys):
        folding.shared_alignment_warning(
            complex_chains(a="s3://bucket/a.a3m", b="s3://bucket/b.a3m")
        )
        assert capsys.readouterr().out == ""


class TestEnvelope:
    def envelope(self, models=None, **extra):
        return folding.envelope(
            name="job",
            predictor="boltz",
            version="2.2.1",
            complex_chains=complex_chains(),
            models=models if models is not None else [{"model": 0}],
            **extra,
        )

    def test_standard_fields(self):
        result = self.envelope()
        assert result["predictor"] == "boltz"
        assert result["version"] == "2.2.1"
        assert result["chain_lengths"] == {"A": len(TARGET), "B": len(BINDER)}
        assert result["chains"]["A"]["sequence"] == TARGET

    def test_best_is_the_first_model(self):
        result = self.envelope(models=[{"model": 1}, {"model": 0}])
        assert result["best"] == {"model": 1}

    def test_no_models_is_not_a_crash(self):
        assert self.envelope(models=[])["best"] is None

    def test_predictor_specific_fields_are_kept(self):
        assert self.envelope(checkpoint="biohub/ESMFold2")["checkpoint"] == (
            "biohub/ESMFold2"
        )


class TestSummaryLines:
    def test_renders_any_predictors_result(self):
        result = folding.envelope(
            name="job",
            predictor="protenix",
            version="2.0.0",
            complex_chains=complex_chains(),
            models=[
                {
                    "model": 0,
                    "structure_uri": "s3://b/model_0.cif",
                    "iptm": 0.91,
                    "complex_plddt": 0.675,
                }
            ],
        )
        (line,) = folding.summary_lines(result)
        assert "iptm=0.910" in line
        assert "plddt=0.675" in line
        assert "s3://b/model_0.cif" in line

    def test_no_models(self):
        assert folding.summary_lines({"models": []}) == ["no models predicted"]


class TestOneServerResolvesNames:
    """A bare service name must reach a predictor as an address.

    Handing Boltz the literal word `colabfold` fails inside its MSA client and
    surfaces as an empty predictions directory with no mention of a URL, which
    is why this is pinned.
    """

    def test_bare_name_becomes_a_url(self):
        chains = complex_chains(a="colabfold")
        assert folding.one_server(chains, "Boltz").startswith("https://")

    def test_a_url_is_unchanged(self):
        chains = complex_chains(a="https://api.colabfold.com")
        assert folding.one_server(chains, "Boltz") == "https://api.colabfold.com"

    def test_a_name_and_its_url_are_one_server(self):
        chains = chain_spec.parse({
            "A": {"sequence": TARGET, "msa": "colabfold"},
            "B": {"sequence": BINDER, "msa": "https://api.colabfold.com"},
        })
        assert folding.one_server(chains, "Boltz") == "https://api.colabfold.com"
