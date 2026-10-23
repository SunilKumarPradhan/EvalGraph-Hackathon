"""Repository URL validation (cloning lives in understanding/buffer.py)."""

from __future__ import annotations

import os
import re
from urllib.parse import urlparse


ALLOWED_HOSTS = {"github.com", "gitlab.com", "bitbucket.org"}


class SourceError(ValueError):
    pass


TREE = re.compile(r"^(?P<repo>/[\w.\-]+/[\w.\-]+?)(?:\.git)?/(?:-/)?tree/(?P<ref>[^/]+)(?:/(?P<sub>.+?))?/?$")


def split_tree_url(url: str) -> tuple[str, str | None, str | None]:
    """https://github.com/o/r/tree/<branch>/<sub/folder> -> (repo url, branch, sub folder). GitLab's /-/tree/ too.
    A branch name containing "/" is read as branch + folder; pass the plain repo URL for such branches."""
    parsed = urlparse(url.strip())
    m = TREE.match(parsed.path)
    if not m:
        return url.strip(), None, None
    sub = m.group("sub")
    if sub and (".." in sub.split("/") or sub.startswith("/")):
        raise SourceError("sub folder must stay inside the repository")
    return f"{parsed.scheme}://{parsed.netloc}{m.group('repo')}", m.group("ref"), sub


def validate_url(url: str) -> str:
    parsed = urlparse(url.strip())
    if parsed.scheme not in {"https", "http"} or not parsed.netloc:
        raise SourceError("repo_url must be an http(s) URL")
    extra = {h.strip().lower() for h in os.getenv("EVALGRAPH_ALLOWED_GIT_HOSTS", "").split(",") if h.strip()}
    if parsed.netloc.lower() not in ALLOWED_HOSTS | extra:
        raise SourceError(f"host {parsed.netloc} is not allowed (allowed: {', '.join(sorted(ALLOWED_HOSTS | extra))})")
    if TREE.match(parsed.path):
        return f"https://{parsed.netloc}{parsed.path.rstrip('/')}"
    if not re.fullmatch(r"/[\w.\-]+/[\w.\-]+(\.git)?/?", parsed.path):
        raise SourceError("repo_url must look like https://host/owner/repo (or .../tree/<branch>/<folder>)")
    return f"https://{parsed.netloc}{parsed.path.rstrip('/')}"
