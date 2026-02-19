"""
Module for computing cryptographic and non-cryptographic hash digests of files and byte buffers.

This module provides utilities for hashing data using various algorithms including xxHash3
(xxh3_64, xxh3_128) for high-performance hashing and standard algorithms (sha256, md5).
It supports hashing both in-memory byte buffers and files on disk, with compatibility for
both standard pathlib.Path and upath.UPath objects.

Supported hash algorithms:
    - xxh3_64: 64-bit xxHash3 (default, fastest)
    - xxh3_128: 128-bit xxHash3
    - sha256: SHA-256 cryptographic hash
    - md5: MD5 hash (legacy, not cryptographically secure)

Example:
    >>> # Hash a file using the default algorithm
    >>> digest = file_digest(Path("myfile.txt"))
    >>>
    >>> # Compute buffer hash with explicit algorithm
    >>> data = b"Hello, World!"
    >>> hex_hash = buffer_digest(data, algorithm="sha256")
    >>>
    >>> # Use with remote paths via UPath
    >>> remote_file = UPath("s3://bucket/file.bin")
    >>> digest = file_digest(remote_file, algorithm="xxh3_128")

"""

import hashlib
from pathlib import Path
from typing import Literal

import xxhash
from upath import UPath

HashAlgorithm = Literal["xxh3_64", "xxh3_128", "sha256", "md5"]


def _get_hasher(algorithm: HashAlgorithm):
    """Get a hasher instance for the specified algorithm.

    Args:
        algorithm: The hashing algorithm to use.

    Returns:
        A hasher instance (hashlib-compliant).
    """
    if algorithm == "xxh3_64":
        return xxhash.xxh3_64()
    elif algorithm == "xxh3_128":
        return xxhash.xxh3_128()
    elif algorithm == "sha256":
        return hashlib.sha256()
    elif algorithm == "md5":
        return hashlib.md5()
    else:
        raise ValueError(f"Unsupported hash algorithm: {algorithm}")


def buffer_digest(input: bytes, algorithm: HashAlgorithm = "xxh3_64") -> str:
    """Compute the digest of a byte buffer.

    Args:
        input: The byte buffer to hash.
        algorithm: The hashing algorithm to use.

    Returns:
        The hexadecimal representation of the buffer's digest.
    """
    hasher = _get_hasher(algorithm)
    hasher.update(input)
    return hasher.hexdigest()


def file_digest(file_path: Path | UPath, algorithm: HashAlgorithm = "xxh3_64") -> str:
    """Compute the digest of a file.

    Args:
        file_path: The path to the file.
        algorithm: The hashing algorithm to use.

    Returns:
        The hexadecimal representation of the file's digest.
    """
    hasher = _get_hasher(algorithm)
    with file_path.open("rb") as f:
        while chunk := f.read(8192):
            hasher.update(chunk)
    return hasher.hexdigest()


if __name__ == "__main__":
    import sys

    # Test the function with itself or a specified file
    if len(sys.argv) > 1:
        test_file = Path(sys.argv[1])
    else:
        # Default to computing digest of this file itself
        test_file = Path(__file__)

    if not test_file.exists():
        print(f"Error: File '{test_file}' does not exist.")
        sys.exit(1)

    print(f"Computing digests for: {test_file}")
    print("-" * 60)

    # Test all algorithms
    for algo in ["xxh3_64", "xxh3_128", "sha256", "md5"]:
        digest = file_digest(test_file, algorithm=algo)
        print(f"{algo:12s}: {digest}")

    # Test with UPath (using default algorithm)
    print("\n" + "-" * 60)
    print("Testing UPath compatibility (default algorithm):")
    upath_file = UPath(test_file)
    path_digest = file_digest(test_file)
    upath_digest = file_digest(upath_file)
    print(f"Path digest:  {path_digest}")
    print(f"UPath digest: {upath_digest}")
    print(f"Digests match: {path_digest == upath_digest}")
