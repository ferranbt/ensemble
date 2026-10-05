"""Render a structure to a PNG, as a Modal action.

One action, `render_structure`. Every other tool here answers with numbers, and
a number can be right about the wrong thing. A self-consistency RMSD of 0.49A
says a prediction returned to its backbone; it does not say the backbone is a
sensible protein rather than a tangle of helices with no core. Looking is the
cheapest check there is and the only one that catches what the metrics were not
asked about.

PyMOL comes from conda-forge, which is the packaging taken from
hgbrian/biomodals: the pip distribution of PyMOL is not the open-source one,
and the conda build needs `libgl1` present even when rendering headless.

    modal deploy tools/render/app.py

    modal run tools/render/app.py::render_structure \\
        --structure-uri s3://bucket/design.pdb
"""

from __future__ import annotations

from typing import Any

import modal

from tools.common.images import STORAGE_PACKAGES, with_local_sources
from tools.common.tool import app_for, tool

TIMEOUT = 15 * 60

# Three-quarter turns about the vertical axis. One view of a protein hides
# whatever is behind it, and a fold cannot be judged from a single angle.
DEFAULT_VIEWS = ((0, 0, 0), (0, 90, 0), (0, 180, 0), (0, 270, 0))

# Distinct per chain, so a target and its binder are told apart at a glance.
CHAIN_COLOURS = ("skyblue", "salmon", "palegreen", "wheat", "lightpink")

app = app_for("render")

image = with_local_sources(
    modal.Image.micromamba(python_version="3.11")
    .micromamba_install("pymol-open-source==2.5.0", channels=["conda-forge"])
    # PyMOL links against libGL even when it never opens a window.
    .apt_install("libgl1")
    .pip_install(*STORAGE_PACKAGES)
)


@tool("render/render_structure", image=image, timeout=TIMEOUT)
def render_structure(
    rt,
    structure_uri: str,
    views: list | None = None,
    width: int = 1200,
    height: int = 900,
    style: str = "cartoon",
    spectrum: bool = False,
    name: str = "render",
) -> dict[str, Any]:
    """Draw a structure and store the images.

    Args:
        structure_uri: `s3://bucket/key` of the structure, `.pdb` or `.cif`.
        views: Rotations to render, each `(x, y, z)` in degrees applied after
            PyMOL orients the structure. Defaults to four views a quarter turn
            apart, because one angle hides whatever is behind it.
        width: Image width in pixels.
        height: Image height in pixels.
        style: `cartoon` shows secondary structure and is what you want for
            judging a fold; `surface` shows shape and is what you want for
            judging a pocket or an interface; `sticks` shows every atom.
        spectrum: Colour along the chain from blue at the N terminus to red at
            the C terminus instead of by chain. Useful for a monomer, where
            per-chain colouring conveys nothing, and for seeing how the chain
            threads through its own fold.
        name: Identifier for the run.

    Returns:
        `images`, one `(view, uri)` per rendering, and `chains` as PyMOL read
        them.
    """
    from pymol import cmd

    local = rt.fetch(structure_uri, "structure")

    cmd.reinitialize()
    cmd.load(str(local))
    cmd.hide("everything")
    cmd.show(style)
    if style == "cartoon":
        # Without this a backbone-only model, which is what a generator
        # produces, renders as nothing at all: PyMOL needs side chains to
        # infer secondary structure, so it is assigned from geometry instead.
        cmd.dss()
        cmd.show("cartoon")

    chains = list(cmd.get_chains())
    if spectrum:
        cmd.spectrum("count", "rainbow", "name CA")
    else:
        for index, chain in enumerate(chains):
            cmd.color(CHAIN_COLOURS[index % len(CHAIN_COLOURS)], f"chain {chain}")

    cmd.bg_color("white")
    cmd.set("ray_opaque_background", "on")
    cmd.set("antialias", "2")
    cmd.set("orthoscopic", "on")
    cmd.orient()

    images = []
    for index, view in enumerate(views or DEFAULT_VIEWS):
        x, y, z = view
        cmd.reset()
        cmd.orient()
        for axis, angle in (("x", x), ("y", y), ("z", z)):
            if angle:
                cmd.rotate(axis, angle)
        image = rt.work / f"{name}_{index}.png"
        cmd.png(str(image), width=width, height=height, dpi=150, ray=1)
        images.append({
            "view": [x, y, z],
            "uri": rt.storage.upload(image, rt.prefix(image.name)),
        })

    return {
        "name": name,
        "structure_uri": structure_uri,
        "style": style,
        "chains": chains,
        "images": images,
    }
