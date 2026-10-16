#!/usr/bin/env python3
"""Download the source dependencies used by the LIBERO-Pro demo."""
import argparse
import shutil
import tempfile
import urllib.request
import zipfile
from pathlib import Path

SOURCES = {
    "LIBERO-PRO": ("uynitsuj/LIBERO-PRO", "47aaa8038930bcdc84ab9ea2867e2ffc8039ab4a"),
    "robosuite": ("Max-Fu/robosuite", "a498b087d4bc5a3981e3d27030d09bc537a537f3"),
    "sam3": ("facebookresearch/sam3", "6fe87d64a5beb9084923d7a9e002741178635b09"),
    "contact_graspnet_pytorch": ("elchun/contact_graspnet_pytorch", "7927318e6ac7055bf1a5f67261587983c13d80c7"),
    "pyroki": ("chungmin99/pyroki", "388e43e1fc0d0ee382968d3dd72970fd62a0450c"),
    "jaxls": ("brentyi/jaxls", "50a58be88c5ef74532f09e3f55268b4f02c490e3"),
}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=Path("external"))
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    for name, (repo, revision) in SOURCES.items():
        target = args.output / name
        if target.exists():
            print(f"Already present: {target}")
            continue
        url = f"https://codeload.github.com/{repo}/zip/{revision}"
        print(f"Downloading {name} ({revision[:12]})", flush=True)
        with tempfile.TemporaryDirectory(dir=args.output) as temporary:
            work = Path(temporary)
            archive = work / "source.zip"
            urllib.request.urlretrieve(url, archive)
            with zipfile.ZipFile(archive) as zf:
                for entry in zf.infolist():
                    resolved = (work / entry.filename).resolve()
                    if not resolved.is_relative_to(work.resolve()):
                        raise ValueError("Invalid archive path")
                zf.extractall(work)
            source = next(p for p in work.iterdir() if p.is_dir())
            shutil.move(str(source), target)
    # Use the downloaded solver for PyRoKi's dependency, with no VCS install.
    metadata = args.output / "pyroki/pyproject.toml"
    if metadata.is_file():
        text = metadata.read_text()
        text = text.replace("jaxls @ git+https://github.com/brentyi/jaxls.git", "jaxls")
        metadata.write_text(text)


if __name__ == "__main__":
    main()
