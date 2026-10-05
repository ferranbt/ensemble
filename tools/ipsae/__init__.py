"""Interface quality scoring for predicted complexes, via ipsae.py.

Plain functions rather than Modal actions: this is numpy work on a single
matrix, cheap enough to run anywhere, though it still exchanges artifacts by
S3 URI like the rest of the tools.
"""
