"""Rosetta's energy function on GPU, via tmol.

The only non-neural score here. Every other metric in this repo is deep
learning trained on overlapping data, so agreement between them is weaker
evidence than it looks; an energy function is wrong for unrelated reasons.
"""
