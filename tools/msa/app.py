"""Multiple sequence alignment search, as a Modal action.

One action, `search_msa`: given a complex's chains, fetch an alignment for each
and store it as an `.a3m` in S3. Boltz and Protenix both accept precomputed
alignments, so this turns the largest accuracy lever those models have into a
step of its own rather than a flag on each of them.

**Results are cached by content in a Volume.** A campaign folds one target
hundreds of times and the alignment is identical every time, so without a cache
this would queue hundreds of identical searches against a server we do not run.
With it, the second call is a file read.

    modal run tools/msa/app.py::search_msa \\
        --chains '{"A": "MKT...", "B": "GHN..."}'
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

from tools.common import sequences as seq_utils
from tools.common.chains import Chain
from tools.common.images import base_image, with_local_sources
from tools.common.tool import app_for, tool
from tools.msa import cache

COLABFOLD_URL = "https://api.colabfold.com"
COLABFOLD_VERSION = "1.6.2"

# The server asks callers to identify themselves, and warns that an absent
# agent will become an error. Add a contact address here if you want them to be
# able to reach you about a problem.
USER_AGENT = "friday-tools/0.1"

# No GPU: this waits on a remote search. Queues on the public server can be
# long, so the timeout is generous.
TIMEOUT = 2 * 60 * 60

CACHE = "msa-cache"

app = app_for("msa")

image = with_local_sources(
    base_image().pip_install(f"colabfold=={COLABFOLD_VERSION}")
)

@tool(
    "msa/search_msa",
    image=image,
    timeout=TIMEOUT,
    cache=CACHE,
    cache_path=cache.CACHE_DIR,
)
def search_msa(
    rt,
    chains: dict[str, dict | str],
    use_env: bool = True,
    use_filter: bool = True,
    pair: bool = False,
    pairing_strategy: str = "greedy",
    server_url: str = COLABFOLD_URL,
    refresh: bool = False,
    max_searches: int = 10,
    name: str = "msa",
) -> dict[str, Any]:
    """Fetch an alignment for each chain of a complex.

    Args:
        chains: Chain id to one-letter sequence. Chains sharing a sequence are
            searched once, so a homodimer costs one query.
        use_env: Include the environmental databases, which is ColabFold's
            default and finds more remote homologues.
        use_filter: Apply the server's diversity filter. Off returns a larger,
            noisier alignment.
        pair: Return a paired alignment instead of per-chain ones. Needed when
            a predictor wants inter-chain pairing for a complex; leave off for
            the per-chain alignments Boltz and Protenix take by default.
        pairing_strategy: `greedy` or `complete`, when pairing.
        server_url: Which alignment server to query. The default is a public
            service, so **every sequence in the request leaves this
            container**. Point it at your own instance to avoid that.
        refresh: Ignore the cache and search again.
        max_searches: Refuse to run more than this many distinct searches in
            one call. A guard against accidentally sending a large batch to a
            shared public server. Cache hits do not count against it, so a
            re-run of the same complex is never blocked.
        name: Identifier for the run.

    Returns:
        `chains`, the same complex with each chain's `msa` now pointing at the
        `s3://` alignment that was found. That is a predictor's `chains`
        parameter, so a fold step consumes this output unchanged. Alongside it,
        `alignments` carries the depth and cache status per chain, which is for
        judging a result rather than feeding it onward.
    """
    seq_utils.validate_chains(chains)

    search_mode = cache.mode(use_env, use_filter, pair, pairing_strategy)
    grouped = cache.unique(chains)
    cache_dir = Path(cache.CACHE_DIR)
    cache_dir.mkdir(parents=True, exist_ok=True)

    searched: dict[str, dict[str, str]] = {}
    found: dict[str, dict[str, Any]] = {}
    fetched = 0

    for sequence, chain_ids in grouped.items():
        entry = cache_dir / f"{cache.key(sequence, search_mode)}.a3m"
        was_cached = entry.is_file() and not refresh

        if was_cached:
            print(f"cache hit for {len(sequence)} residues", flush=True)
        else:
            if fetched >= max_searches:
                raise ValueError(
                    f"This call needs more than {max_searches} searches "
                    f"({len(grouped)} distinct sequences, {fetched} already "
                    f"fetched). Raise max_searches deliberately if you mean "
                    f"to send that many to {server_url}."
                )
            print(f"searching {search_mode} for {len(sequence)} residues", flush=True)
            entry.write_text(_run_search(
                sequence, rt.work, search_mode, use_env, use_filter,
                pair, pairing_strategy, server_url,
            ))
            fetched += 1

        text = entry.read_text()
        uri = rt.storage.upload(entry, rt.prefix(f"{chain_ids[0]}.a3m"))
        for chain_id in chain_ids:
            searched[chain_id] = Chain(sequence=sequence, msa=uri).as_dict()
            found[chain_id] = {
                "num_sequences": text.count(">"),
                "cached": was_cached,
            }

    # Persist anything this call just fetched for the next one.
    if fetched:
        rt.save_cache()

    return {
        "name": name,
        "mode": search_mode,
        "server_url": server_url,
        "searched": fetched,
        "reused": len(grouped) - fetched,
        "chains": searched,
        "alignments": found,
    }


def _run_search(
    sequence: str,
    work: Path,
    search_mode: str,
    use_env: bool,
    use_filter: bool,
    pair: bool,
    pairing_strategy: str,
    server_url: str,
) -> str:
    """One query against the alignment server, returning a3m text.

    The client handles submission, polling, rate limits and backoff, and
    returns one a3m per query sequence.
    """
    from colabfold.colabfold import run_mmseqs2

    result = run_mmseqs2(
        [sequence],
        prefix=str(work / cache.key(sequence, search_mode)),
        use_env=use_env,
        use_filter=use_filter,
        use_templates=False,
        use_pairing=pair,
        pairing_strategy=pairing_strategy,
        host_url=server_url,
        user_agent=USER_AGENT,
    )
    # The client returns a3m lines, and a second element for templates when
    # those were requested. Only the alignments are wanted here.
    lines = result[0] if isinstance(result, tuple) else result
    if not lines:
        raise RuntimeError(
            f"The alignment server returned nothing for a {len(sequence)} "
            f"residue query in mode {search_mode}."
        )
    return lines[0]
