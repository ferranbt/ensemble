"""Structural alignment and TM-score, via US-align.

Answers "are these the same fold", which RMSD cannot: TM-score is
length-normalised, so it is comparable across proteins of different sizes,
where a fixed RMSD in Angstroms is not.
"""
