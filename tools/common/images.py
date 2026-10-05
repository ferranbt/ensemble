"""Shared Modal image building blocks.

Every tool in this package runs a third-party CLI inside a container that also
needs to import `tools.common`, so the base image pins the Python version and
installs git for source checkouts.
"""

from __future__ import annotations

import modal

PYTHON_VERSION = "3.11"

SECRETS = [
    modal.Secret.from_name("aws-secret"),
    modal.Secret.from_name("friday-s3"),
    modal.Secret.from_name("friday-postgres"),
]

# Local Python sources every tool container needs, as importable module names.
LOCAL_SOURCES = ("tools",)

# Every tool moves artifacts through S3, may be handed a structure format it
# does not read, and records what it did, so all three belong in the base
# image. psycopg is inert without DATABASE_URL set.
STORAGE_PACKAGES = ("boto3==1.35.*", "gemmi==0.7.5", "psycopg[binary]==3.2.*")


def base_image(python_version: str = PYTHON_VERSION) -> modal.Image:
    """Debian slim with git for source checkouts and boto3 for artifacts."""
    return (
        modal.Image.debian_slim(python_version=python_version)
        .apt_install("git")
        .pip_install(*STORAGE_PACKAGES)
    )


def torch_image(
    torch_version: str = "2.4.1",
    numpy_version: str = "1.26.4",
    python_version: str = PYTHON_VERSION,
) -> modal.Image:
    """Base image plus CUDA-enabled PyTorch from the standard PyPI wheel."""
    return base_image(python_version).pip_install(
        f"torch=={torch_version}", f"numpy=={numpy_version}"
    )


def with_local_sources(image: modal.Image) -> modal.Image:
    """Make the local `tools` package importable. Must be the final build step."""
    return image.add_local_python_source(*LOCAL_SOURCES)
