"""ESM-2 variant effect scoring as Modal GPU actions.

Two actions, split by what they answer:

    scan_mutations   where should I mutate, and to what
    score_variants   how good are these specific candidates

ESM-2 sees sequence only, with no structure. That is the point. ProteinMPNN's
`residue_probabilities` already tells you what the backbone tolerates, and this
tells you what evolution favours. They are independent signals, and mutations
both like are the ones worth making.

Scores are log likelihood ratios: `log p(mutant) - log p(wild type)` at the
mutated position. Positive means the model prefers the substitution. They are
relative, not calibrated, so compare them within one protein and do not read an
absolute threshold into them.

    modal deploy tools/esm/app.py

    modal run tools/esm/app.py::scan_mutations --sequence MKTAYIAK...
"""

from __future__ import annotations

from typing import Any

from tools.common import mutations as mut
from tools.common.images import torch_image, with_local_sources
from tools.common.tool import app_for, tool

# ESM-2 checkpoints, smallest to largest. The 650M model is the usual choice:
# the 3B is better by a small margin for several times the cost.
MODELS = {
    "8M": "facebook/esm2_t6_8M_UR50D",
    "35M": "facebook/esm2_t12_35M_UR50D",
    "150M": "facebook/esm2_t30_150M_UR50D",
    "650M": "facebook/esm2_t33_650M_UR50D",
    "3B": "facebook/esm2_t36_3B_UR50D",
}

TRANSFORMERS_VERSION = "4.44.2"

GPU = "A10G"
TIMEOUT = 30 * 60
CACHE = "esm-cache"
CACHE_DIR = "/cache/huggingface"

app = app_for("esm")

image = with_local_sources(
    torch_image()
    .pip_install(f"transformers=={TRANSFORMERS_VERSION}")
    .env({"HF_HOME": CACHE_DIR})
)

OPTIONS = dict(
    image=image, gpu=GPU, timeout=TIMEOUT, cache=CACHE, cache_path=CACHE_DIR
)

def _load(rt, model: str):
    """Load one ESM-2 checkpoint onto the GPU, in evaluation mode."""
    import torch
    from transformers import AutoTokenizer, EsmForMaskedLM

    if model not in MODELS:
        raise ValueError(
            f"Unknown model {model!r}. Choose from {', '.join(MODELS)}"
        )
    name = MODELS[model]
    tokenizer = AutoTokenizer.from_pretrained(name)
    network = EsmForMaskedLM.from_pretrained(name).eval().cuda()
    rt.save_cache()
    torch.set_grad_enabled(False)
    return tokenizer, network


def _log_probs(rt, sequence: str, model: str, scoring: str, batch_size: int):
    """Per-position log probabilities over the amino acid vocabulary.

    Returns an array of shape (length, 20) aligned to `mutations.RESIDUES`,
    plus the loaded tokenizer so callers can map residues to token ids.

    Two scorings, following the ESM variant-effect literature:

    `masked-marginals` masks each position in turn and reads the distribution
    the model puts there without seeing the true residue. It costs one forward
    pass per position and is the more faithful of the two.

    `wt-marginals` reads every position from a single pass over the unmasked
    sequence. It is length-times cheaper and less accurate, because the model
    can see the residue it is being asked to score.
    """
    import torch

    tokenizer, network = _load(rt, model)
    encoded = tokenizer(sequence, return_tensors="pt")
    tokens = encoded["input_ids"].cuda()
    # The tokenizer prepends a start token, so sequence position i is token i+1.
    offset = 1
    length = len(sequence)
    residue_ids = torch.tensor(
        [tokenizer.convert_tokens_to_ids(a) for a in mut.RESIDUES], device="cuda"
    )

    if scoring == "wt-marginals":
        logits = network(tokens).logits[0]
        table = torch.log_softmax(logits, dim=-1)[offset:offset + length]
        return table[:, residue_ids].float().cpu().numpy(), tokenizer

    if scoring != "masked-marginals":
        raise ValueError(
            f"scoring must be 'masked-marginals' or 'wt-marginals', "
            f"got {scoring!r}"
        )

    rows = []
    for start in range(0, length, batch_size):
        positions = range(start, min(start + batch_size, length))
        batch = tokens.repeat(len(positions), 1)
        for row, position in enumerate(positions):
            batch[row, position + offset] = tokenizer.mask_token_id
        logits = network(batch).logits
        for row, position in enumerate(positions):
            table = torch.log_softmax(logits[row, position + offset], dim=-1)
            rows.append(table[residue_ids].float().cpu().numpy())
        print(f"scored {min(start + batch_size, length)}/{length}", flush=True)

    import numpy as np

    return np.stack(rows), tokenizer


@tool("esm/scan_mutations", **OPTIONS)
def scan_mutations(
    rt,
    sequence: str,
    positions: list[int] | None = None,
    model: str = "650M",
    scoring: str = "masked-marginals",
    top_k: int = 5,
    batch_size: int = 8,
    name: str = "scan",
) -> dict[str, Any]:
    """Score every substitution at every position, or at chosen positions.

    Use this to decide where to mutate. The full matrix goes to S3; the return
    carries the best substitutions per position so an agent can act without
    downloading anything.

    Args:
        sequence: The protein to scan, one letter per residue.
        positions: 1-indexed positions to report. Empty means every position.
            Note that `masked-marginals` still runs the whole sequence, since
            each position needs its own forward pass either way.
        model: One of `MODELS`, by size. 650M is the usual choice.
        scoring: `masked-marginals` or `wt-marginals`. See `_log_probs`.
        top_k: How many substitutions to report per position, best first.
        batch_size: Masked positions per forward pass. Lower it on memory errors.
        name: Identifier for the run.

    Returns:
        `positions`, one entry per reported position with the wild-type
        residue, the best substitutions by log likelihood ratio, and the
        entropy of the distribution. `matrix_uri` points at the full table.
        A high ratio means the model prefers the substitution to what is there.
    """
    import numpy as np

    sequence = mut.clean_sequence(sequence)

    table, _ = _log_probs(rt, sequence, model, scoring, batch_size)
    wt_index = np.array([mut.RESIDUES.index(a) for a in sequence])
    wt_logp = table[np.arange(len(sequence)), wt_index]
    ratios = table - wt_logp[:, None]

    probabilities = np.exp(table)
    probabilities /= probabilities.sum(axis=1, keepdims=True)
    with np.errstate(divide="ignore", invalid="ignore"):
        entropy = -(probabilities * np.where(
            probabilities > 0, np.log2(probabilities), 0.0)).sum(axis=1)

    archive = rt.work / "mutation_scan.npz"
    np.savez_compressed(
        archive,
        log_likelihood_ratio=ratios.astype(np.float32),
        log_probs=table.astype(np.float32),
        residues=np.array(list(mut.RESIDUES)),
        sequence=sequence,
    )

    wanted = sorted(set(positions)) if positions else range(1, len(sequence) + 1)
    reported: list[dict[str, Any]] = []
    for position in wanted:
        if not 1 <= position <= len(sequence):
            raise ValueError(
                f"Position {position} is outside the sequence, which is "
                f"{len(sequence)} residues long"
            )
        i = position - 1
        order = np.argsort(-ratios[i])
        reported.append({
            "position": position,
            "wild_type": sequence[i],
            "entropy_bits": float(entropy[i]),
            "top": [
                {
                    "mutation": f"{sequence[i]}{position}{mut.RESIDUES[j]}",
                    "aa": mut.RESIDUES[j],
                    "log_likelihood_ratio": float(ratios[i, j]),
                }
                for j in order[:top_k] if mut.RESIDUES[j] != sequence[i]
            ],
        })

    return {
        "name": name,
        "model": model,
        "scoring": scoring,
        "length": len(sequence),
        "matrix_uri": rt.storage.upload(archive, rt.prefix("mutation_scan.npz")),
        "positions": reported,
    }


@tool("esm/score_variants", **OPTIONS)
def score_variants(
    rt,
    sequence: str,
    variants: list[str],
    model: str = "650M",
    scoring: str = "masked-marginals",
    batch_size: int = 8,
    name: str = "variants",
) -> dict[str, Any]:
    """Score specific variants, including ones carrying several mutations.

    Use this to rank candidates you already have, for instance the survivors of
    a scan or designs proposed elsewhere.

    Args:
        sequence: The wild-type sequence the variants are called against.
        variants: Variants in `A123K` notation, several mutations joined by
            `:`, for example `["A50K", "T54R:K57A"]`.
        model: One of `MODELS`, by size.
        scoring: `masked-marginals` or `wt-marginals`.
        batch_size: Masked positions per forward pass.
        name: Identifier for the run.

    Returns:
        `scored`, one entry per variant sorted best first, carrying the summed
        log likelihood ratio and the per-mutation contributions. A variant's
        score is the sum of its mutations', which assumes they act
        independently. That assumption weakens as mutations get closer together
        in the structure, so treat multi-mutant scores as a rough ordering.
    """
    import numpy as np

    sequence = mut.clean_sequence(sequence)

    parsed = {variant: mut.parse(variant) for variant in variants}
    for variant, changes in parsed.items():
        mut.validate(changes, sequence)

    table, _ = _log_probs(rt, sequence, model, scoring, batch_size)
    wt_index = np.array([mut.RESIDUES.index(a) for a in sequence])
    ratios = table - table[np.arange(len(sequence)), wt_index][:, None]

    scored: list[dict[str, Any]] = []
    for variant, changes in parsed.items():
        contributions = [
            {
                "mutation": str(change),
                "log_likelihood_ratio": float(
                    ratios[change.index, mut.RESIDUES.index(change.mt)]
                ),
            }
            for change in changes
        ]
        scored.append({
            "variant": variant,
            "sequence": mut.apply(sequence, changes),
            "num_mutations": len(changes),
            "score": sum(c["log_likelihood_ratio"] for c in contributions),
            "mutations": contributions,
        })

    scored.sort(key=lambda entry: entry["score"], reverse=True)
    return {
        "name": name,
        "model": model,
        "scoring": scoring,
        "length": len(sequence),
        "scored": scored,
    }
