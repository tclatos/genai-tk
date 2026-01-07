from pathlib import Path

import xxhash
from upath import UPath


def file_digest(file_path: Path | UPath) -> str:
    """Compute the xxHash64 digest of a file.

    Args:
        file_path (str | Path | UPath): The path to the file.

    Returns:
        str: The hexadecimal representation of the file's xxHash64 digest.
    """
    hasher = xxhash.xxh3_64()
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

    print(f"Computing xxHash64 digest for: {test_file}")
    digest = file_digest(test_file)
    print(f"Digest: {digest}")

    # Test with UPath
    upath_file = UPath(test_file)
    upath_digest = file_digest(upath_file)
    print("\nUsing UPath:")
    print(f"Digest: {upath_digest}")
    print(f"Digests match: {digest == upath_digest}")
