"""Download the immutable public CZ stock-level panel and verify its LFS hash."""
from pathlib import Path
import hashlib
import json
import os
import time
from urllib.parse import quote

HERE = Path(__file__).resolve().parent
SOURCE = HERE.parents[2] / "Chen_Zimmermann_Factor_Zoo_2026_10"
RELEASE = "Data-Release-2026.10"
FILENAME = "Firm Level Characteristics/Predictors.parquet"


def download():
    import requests
    folder = SOURCE / "Firm Level Characteristics"
    folder.mkdir(exist_ok=True)
    target = folder / "Predictors.parquet"
    receipt = folder / "download_receipt.json"
    session = requests.Session()
    metadata_url = f"https://huggingface.co/api/datasets/OpenSourceAP/CrossSection/tree/{RELEASE}/Firm%20Level%20Characteristics"
    response = session.get(metadata_url, timeout=60)
    response.raise_for_status()
    meta = next(row for row in response.json() if row["path"] == FILENAME)
    expected = meta["lfs"]["oid"]
    size = meta["size"]
    url = f"https://huggingface.co/datasets/OpenSourceAP/CrossSection/resolve/{RELEASE}/{quote(FILENAME)}"
    if target.exists():
        with target.open("rb") as handle:
            assert hashlib.file_digest(handle, "sha256").hexdigest() == expected
    else:
        partial = target.with_suffix(".partial")
        for attempt in range(6):
            start = partial.stat().st_size if partial.exists() else 0
            if start == size:
                break
            try:
                headers = {"Range": f"bytes={start}-"} if start else {}
                # A fresh URL prevents expired cached signed Xet URLs on resume.
                with session.get(url, params={"download": "true", "task_time": int(time.time())},
                                 headers=headers, stream=True, timeout=(30, 180)) as result:
                    result.raise_for_status()
                    append = bool(start and result.status_code == 206)
                    with partial.open("ab" if append else "wb") as handle:
                        for chunk in result.iter_content(8 * 1024 * 1024):
                            handle.write(chunk)
                    if partial.stat().st_size != size:
                        raise IOError(f"Incomplete download: {partial.stat().st_size}/{size}")
                break
            except requests.RequestException:
                if attempt == 5:
                    raise
                time.sleep(3)
        with partial.open("rb") as handle:
            actual = hashlib.file_digest(handle, "sha256").hexdigest()
        assert partial.stat().st_size == size and actual == expected, "CZ checksum mismatch"
        os.replace(partial, target)
    receipt.write_text(json.dumps(dict(release=RELEASE, source_url=url, metadata_url=metadata_url,
        bytes=size, sha256=expected, verified=True), indent=2) + "\n")
    print(json.dumps(dict(path=str(target), bytes=size, sha256=expected, verified=True)), flush=True)
    return target


if __name__ == "__main__":
    download()
