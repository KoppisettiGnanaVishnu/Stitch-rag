
import hashlib
import json
import time
from pathlib import Path

import numpy as np


MAX_ENTRIES = 32


def fingerprint_source(source_name, file_bytes, extra=None):
    """
    Stable identity for a document: content hash +
    source name (plus any extra parameters, e.g.
    chunking settings, that change how it is indexed).
    Same file re-uploaded under the same name always
    maps to the same index entry.
    """

    hasher = hashlib.sha256()

    hasher.update(source_name.encode("utf-8"))
    hasher.update(b"\x00")
    hasher.update(file_bytes)

    if extra:
        hasher.update(b"\x00")
        hasher.update(extra)

    return hasher.hexdigest()[:24]


class IndexStore:
    """
    Flat local vector store.

    One compressed NumPy file (embeddings) plus one
    JSON sidecar (chunks) per indexed document, and a
    manifest that tracks entries and recency.

    Deliberately simple and transparent: at Stitch's
    scale (thousands of chunks) a flat NumPy index is
    fast, needs no external service, and keeps
    everything inspectable. The pipeline only ever
    talks to this interface, so it can be swapped for
    FAISS or Qdrant later without touching anything
    else.
    """

    def __init__(self, index_dir):

        self.index_dir = Path(index_dir)

        self.index_dir.mkdir(
            parents=True,
            exist_ok=True
        )

        self.manifest_path = (
            self.index_dir / "manifest.json"
        )

    # ------------------------------------------
    # Manifest helpers
    # ------------------------------------------

    def _manifest(self):

        try:

            return json.loads(
                self.manifest_path.read_text(
                    encoding="utf-8"
                )
            )

        except (OSError, json.JSONDecodeError):

            return {"version": 1, "entries": {}}

    def _write_manifest(self, manifest):

        self.manifest_path.write_text(
            json.dumps(manifest, indent=2),
            encoding="utf-8"
        )

    def _paths(self, entry_hash):

        base = self.index_dir / f"entry_{entry_hash}"

        return (
            Path(str(base) + ".npz"),
            Path(str(base) + ".json"),
        )

    # ------------------------------------------
    # Read / write / prune
    # ------------------------------------------

    def get(self, entry_hash):
        """
        Return (chunks, embeddings) for a document
        fingerprint, or None when absent or corrupt.
        """

        npz_path, json_path = self._paths(entry_hash)

        try:

            chunks = json.loads(
                json_path.read_text(encoding="utf-8")
            )

            with np.load(npz_path) as data:
                embeddings = list(data["embeddings"])

            return chunks, embeddings

        except (OSError, ValueError, KeyError):

            return None

    def put(self, entry_hash, chunks, embeddings):

        npz_path, json_path = self._paths(entry_hash)

        matrix = np.asarray(
            embeddings,
            dtype="float32"
        )

        np.savez_compressed(
            npz_path,
            embeddings=matrix
        )

        json_path.write_text(
            json.dumps(
                chunks,
                ensure_ascii=False
            ),
            encoding="utf-8"
        )

        manifest = self._manifest()

        entries = manifest.setdefault("entries", {})

        entries[entry_hash] = {
            "chunks": len(chunks),
            "last_used": time.time(),
        }

        self._write_manifest(manifest)

    def prune(self, active_hashes):
        """
        Refresh recency for the documents just indexed,
        then evict the oldest entries beyond the cap so
        the store never grows unbounded.
        """

        manifest = self._manifest()

        entries = manifest.setdefault("entries", {})

        for entry_hash in list(entries):

            if entry_hash in active_hashes:

                entries[entry_hash]["last_used"] = (
                    time.time()
                )

        if len(entries) > MAX_ENTRIES:

            ordered = sorted(
                entries.items(),
                key=lambda item: item[1].get("last_used", 0)
            )

            for entry_hash, _ in ordered[
                : len(entries) - MAX_ENTRIES
            ]:

                npz_path, json_path = self._paths(entry_hash)

                for path in (npz_path, json_path):

                    try:

                        path.unlink()

                    except OSError:

                        pass

                entries.pop(entry_hash, None)

        self._write_manifest(manifest)
