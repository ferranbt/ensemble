"""Where a predictor's evolutionary context comes from.

An alignment is the largest accuracy lever these models have and the only
setting that can send your sequences somewhere, so it is one parameter with
three meanings rather than three parameters:

    ""                          single sequence; nothing leaves the container
    "colabfold"                 shorthand for the public ColabFold service
    "https://msa.internal"      any alignment server, including your own
    "s3://bucket/key.a3m"       a precomputed alignment, already searched

The last is what makes searching a step of its own: `tools.msa` produces those
URIs and a predictor consumes them, so one alignment can be reused across
hundreds of folds instead of being re-fetched by each.

A complex can mix them. Passing a mapping sets each chain independently, which
is the normal case for a binder: the target has an alignment worth having and
the de novo binder has no homologues to find.

    msa = "s3://bucket/target.a3m"          # every chain, same alignment
    msa = {"A": "s3://bucket/target.a3m"}   # chain A only; B single sequence
"""

from __future__ import annotations

from urllib.parse import urlparse

COLABFOLD_URL = "https://api.colabfold.com"

# Services a tool may know by name rather than by address.
KNOWN_SERVERS = {
    COLABFOLD_URL: "colabfold",
    "https://api.protenix.ai": "protenix",
}

# What a setting resolves to.
NONE = "none"
SERVER = "server"
ARTIFACT = "artifact"


def normalise(msa: str) -> str:
    """Validate one setting, returning "" for single sequence or a URI.

    Accepts the empty string, `none`/`off`/`empty` for single sequence, the
    bare word `colabfold`, any http(s) server URL, or an `s3://` alignment.
    """
    text = (msa or "").strip().rstrip("/")
    if not text or text.lower() in ("none", "off", "empty", "false"):
        return ""
    if text.lower() == "colabfold":
        return COLABFOLD_URL
    parsed = urlparse(text)
    if parsed.scheme in ("http", "https") and parsed.netloc:
        return text
    if parsed.scheme == "s3" and parsed.netloc and parsed.path.strip("/"):
        return text
    raise ValueError(
        f"msa must be empty for single sequence, 'colabfold', an http(s) "
        f"server URL, or an s3:// alignment, got {msa!r}"
    )


def kind(msa: str) -> str:
    """Whether a setting means no alignment, a server to ask, or a file."""
    uri = normalise(msa)
    if not uri:
        return NONE
    return ARTIFACT if uri.startswith("s3://") else SERVER


def per_chain(
    msa: str | dict[str, str] | None, chains: list[str] | dict[str, str]
) -> dict[str, str]:
    """Expand a setting to one normalised value per chain.

    A single value applies to every chain. A mapping sets chains
    independently, and any chain it omits runs single sequence, which is the
    right default for a designed binder.

    Raises:
        ValueError: if the mapping names a chain that is not in the job, since
            that is a typo whose only symptom would be a silently missing
            alignment.
    """
    chain_ids = list(chains)
    if msa is None or isinstance(msa, str):
        value = normalise(msa or "")
        return {chain_id: value for chain_id in chain_ids}

    unknown = sorted(set(msa) - set(chain_ids))
    if unknown:
        raise ValueError(
            f"msa names chain(s) {unknown} that are not in this job. "
            f"It has: {', '.join(chain_ids)}"
        )
    return {
        chain_id: normalise(msa.get(chain_id, "")) for chain_id in chain_ids
    }


def enabled(msa: str) -> bool:
    """Whether any alignment will be used at all."""
    return kind(msa) != NONE


def servers(settings: dict[str, str]) -> list[str]:
    """The distinct alignment servers a resolved job needs to query."""
    return sorted({v for v in settings.values() if v and not v.startswith("s3://")})


def is_public(msa: str) -> bool:
    """Whether this sends sequences to a service we do not run.

    A private instance is not in `KNOWN_SERVERS`, so this is False for it,
    which is the point. An `s3://` alignment sends nothing anywhere.
    """
    return normalise(msa) in KNOWN_SERVERS


def protenix_mode(msa: str) -> str:
    """Translate a server URL into the named mode Protenix accepts.

    Raises:
        ValueError: for a server Protenix has no name for, rather than quietly
            querying a different one than the caller asked for.
    """
    uri = normalise(msa)
    if not uri:
        return ""
    if uri.startswith("s3://"):
        raise ValueError(
            "A precomputed alignment is passed to Protenix as a file path, "
            "not as a server mode."
        )
    mode = KNOWN_SERVERS.get(uri)
    if mode is None:
        raise ValueError(
            f"Protenix cannot use an arbitrary MSA server. It accepts only "
            f"{', '.join(sorted(set(KNOWN_SERVERS.values())))}, so {uri} is "
            f"not reachable from it. Use Boltz for a custom server, or run "
            f"this call single sequence."
        )
    return mode
