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
