#!/usr/bin/env python3

"""Publish the S2 sea-ice dataset to Zenodo via the REST API.

Browser uploads are unreliable above a few GB, and S2_tiff.zip is 8.3 GB, so
the files go up through the bucket API (streamed, no whole-file buffering).

Typical use:

    # 1. Rehearse on the sandbox (separate account + token from production)
    export ZENODO_TOKEN=<sandbox token>
    ./publish_to_zenodo.py --sandbox --files data/*.zip

    # 2. For real — creates a draft and stops before publishing
    export ZENODO_TOKEN=<production token>
    ./publish_to_zenodo.py --files data/*.zip

    # 3. Review the draft in the browser, then release it
    ./publish_to_zenodo.py --deposition 1234567 --publish

Get a token at Account -> Applications -> Personal access tokens with the
`deposit:write` and `deposit:actions` scopes.

Published files are IMMUTABLE. A mistake cannot be corrected in place; it
requires publishing a new version. Rehearse on --sandbox first.
"""

import argparse
import hashlib
import json
import logging
import os
import sys

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

try:
    import requests
except ImportError:
    logger.error("requests not installed. Run: pip install requests")
    sys.exit(1)

PROD = "https://zenodo.org"
SANDBOX = "https://sandbox.zenodo.org"


def api(base):
    return f"{base}/api"


def check(resp, what):
    """Raise with Zenodo's own error body, which is far more useful than the status line."""
    if resp.status_code not in (200, 201, 202, 204):
        try:
            detail = json.dumps(resp.json(), indent=2)
        except Exception:
            detail = resp.text[:2000]
        logger.error(f"{what} failed [HTTP {resp.status_code}]\n{detail}")
        sys.exit(1)
    return resp


def md5sum(path, chunk=8 << 20):
    h = hashlib.md5()
    with open(path, "rb") as fh:
        for block in iter(lambda: fh.read(chunk), b""):
            h.update(block)
    return h.hexdigest()


def human(n):
    for unit in ("B", "KB", "MB", "GB"):
        if n < 1024 or unit == "GB":
            return f"{n:.1f} {unit}"
        n /= 1024


def create_deposition(base, token, metadata):
    r = check(requests.post(f"{api(base)}/deposit/depositions",
                            params={"access_token": token},
                            json={"metadata": metadata},
                            timeout=60),
              "Create deposition")
    d = r.json()
    logger.info(f"Created draft deposition {d['id']}")
    return d


def upload_file(bucket_url, token, path):
    """Stream one file into the deposition's bucket, verifying the server's checksum."""
    name = os.path.basename(path)
    size = os.path.getsize(path)
    logger.info(f"Uploading {name} ({human(size)}) — this can take a while")

    local = md5sum(path)
    with open(path, "rb") as fh:
        r = check(requests.put(f"{bucket_url}/{name}",
                               params={"access_token": token},
                               data=fh,               # streamed, not buffered
                               timeout=None),
                  f"Upload {name}")
    remote = r.json().get("checksum", "").replace("md5:", "")
    if remote and remote != local:
        logger.error(f"Checksum mismatch for {name}: local {local} != zenodo {remote}")
        sys.exit(1)
    logger.info(f"  ok — md5 {local} verified")


def main():
    p = argparse.ArgumentParser(description="Publish the dataset to Zenodo",
                                formatter_class=argparse.RawDescriptionHelpFormatter,
                                epilog=__doc__)
    p.add_argument("--files", nargs="+", default=[], help="Files to upload")
    p.add_argument("--metadata", default="zenodo_metadata.json",
                   help="Metadata JSON (default: zenodo_metadata.json)")
    p.add_argument("--sandbox", action="store_true",
                   help="Use sandbox.zenodo.org (separate account and token)")
    p.add_argument("--deposition", type=int, default=None,
                   help="Add files to / publish an existing draft instead of creating one")
    p.add_argument("--publish", action="store_true",
                   help="Actually publish. Without this the draft is left for review.")
    args = p.parse_args()

    token = os.environ.get("ZENODO_TOKEN")
    if not token:
        logger.error("ZENODO_TOKEN is not set. Create one at "
                     "Account -> Applications -> Personal access tokens "
                     "(scopes: deposit:write, deposit:actions).")
        sys.exit(1)

    base = SANDBOX if args.sandbox else PROD
    logger.info(f"Target: {base}{'  (SANDBOX — nothing here is permanent)' if args.sandbox else ''}")

    for f in args.files:
        if not os.path.isfile(f):
            logger.error(f"No such file: {f}")
            sys.exit(1)

    # --- draft ---
    if args.deposition:
        r = check(requests.get(f"{api(base)}/deposit/depositions/{args.deposition}",
                               params={"access_token": token}, timeout=60),
                  "Fetch deposition")
        dep = r.json()
    else:
        if not os.path.isfile(args.metadata):
            logger.error(f"Metadata file not found: {args.metadata}")
            sys.exit(1)
        metadata = json.load(open(args.metadata))
        placeholder = [c for c in metadata.get("creators", [])
                       if "REPLACE ME" in c.get("name", "")]
        if placeholder:
            logger.error(f"{args.metadata} still has placeholder creators. "
                         "Fill in the real authors before publishing — this is "
                         "the attribution that appears on the DOI.")
            sys.exit(1)
        dep = create_deposition(base, token, metadata)

    bucket = dep["links"]["bucket"]

    # --- files ---
    for f in args.files:
        upload_file(bucket, token, f)

    # --- publish ---
    if args.publish:
        logger.warning("Publishing is PERMANENT — files can never be changed, only superseded.")
        r = check(requests.post(
            f"{api(base)}/deposit/depositions/{dep['id']}/actions/publish",
            params={"access_token": token}, timeout=120), "Publish")
        out = r.json()
        doi = out.get("doi")
        rec = out.get("record_id", dep["id"])
        logger.info("=" * 68)
        logger.info(f"Published.  DOI: {doi}")
        logger.info(f"Record:     {base}/records/{rec}")
        logger.info("")
        logger.info("Stage it with:")
        logger.info(f"  S2_DATA_URL={base}/records/{rec}/files \\")
        logger.info("  S2_URL_SUFFIX='?download=1' ./prepare_author_data.sh")
        logger.info("=" * 68)
    else:
        logger.info("=" * 68)
        logger.info(f"Draft {dep['id']} ready — NOT published.")
        logger.info(f"Review it at {base}/uploads/{dep['id']}")
        logger.info(f"Then: ./publish_to_zenodo.py {'--sandbox ' if args.sandbox else ''}"
                    f"--deposition {dep['id']} --publish")
        logger.info("=" * 68)


if __name__ == "__main__":
    main()
