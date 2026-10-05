"""Utilities shared by every tool wrapper.

Submodules are imported explicitly rather than re-exported here, so that pure
parsing code (`fasta`, and the tools' own output readers) stays usable without
Modal installed. Only `images` requires it.
"""
