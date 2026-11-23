#!/usr/bin/env python3
"""Regenerate feed.md — the time-ordered feed of findings and ingestions.

Sources (all local, no network):
  * Weekly digest ledger:
    ~/workspace/goals/expert-in-ai-inference-and-training-performance/hidden_files/digest-seen.md
    (sections by Monday date, ~15 "Title — link" items per week; tag "Digest")
  * TPU deep-scan ledger:
    ~/workspace/goals/expert-in-ai-inference-and-training-performance/hidden_files/tpu-scan-seen.md
    (sections by Monday date, prose items with links; tag "TPU scan")
  * Ingest commits on this branch's history (git log --since=2026-09-01):
    commits whose subject starts with Ingest/Add/Adding and which touch
    .md files under a KB section; tag "Ingested", linked to the KB page
    that gained the most lines in that commit.

Rules enforced here:
  * No invented summaries — entries carry the ledger's own title + link.
  * De-duplication: an item already kept under a different tag is dropped.
    Same URL wins for the newest entry; within one week, a TPU-scan item
    whose distinctive title tokens are >=60% contained in a digest item's
    title is treated as the same item (digest version kept). This also
    folds repeated identical links (e.g. the weekly Ironwood pricing
    verification) into their newest occurrence.
  * Entries are single bullets grouped under reverse-chronological
    "Week of YYYY-MM-DD" (Monday-start) headings.

Usage: python3 scripts/build_feed.py [--output feed.md]
"""

import argparse
import re
import subprocess
import sys
from datetime import date, datetime, timedelta
from pathlib import Path
from zoneinfo import ZoneInfo

USER_TZ = ZoneInfo("America/Los_Angeles")

REPO_ROOT = Path(__file__).resolve().parent.parent
GOAL_DIR = (
    Path.home()
    / "workspace/goals/expert-in-ai-inference-and-training-performance/hidden_files"
)
DIGEST_SEEN = GOAL_DIR / "digest-seen.md"
TPU_SEEN = GOAL_DIR / "tpu-scan-seen.md"

INGEST_SINCE = "2026-09-01"  # start of the digest era
SECTION_DIRS = (
    "hardware/",
    "workloads/",
    "modeling/",
    "simulation/",
    "scheduling/",
    "characterization/",
    "references/",
)
INGEST_SUBJECT = re.compile(r"^(Ingest|Add|Adding)\b")
URL_RE = re.compile(r"https?://[^\s)>\]]+")
SECTION_RE = re.compile(r"^## (\d{4}-\d{2}-\d{2})\b")
ITEM_RE = re.compile(r"^(\d+)\.\s+(.*)$")
STOPWORDS = {
    "the", "and", "for", "with", "from", "google", "introducing", "announces",
    "announce", "results", "result", "performance", "training", "inference",
    "new", "now", "open", "source", "run", "using", "into", "more", "faster",
}
TITLE_CAP = 160
OVERLAP_THRESHOLD = 0.6


def clean_url(raw: str):
    """Return a usable URL, or None if the ledger's URL is truncated."""
    if raw.endswith("..."):
        return None  # truncated in the ledger; do not link a dead prefix
    return raw.rstrip(".,;")


def norm_url(url: str) -> str:
    return url.lower().rstrip("/")


def monday_of(d: date) -> date:
    return d - timedelta(days=d.weekday())


def parse_ledger(path: Path, tag: str):
    """Parse a seen-list ledger into (week_date, title, [urls]) entries."""
    entries = []
    week = None
    dropped_truncated = 0
    for line in path.read_text().splitlines():
        m = SECTION_RE.match(line)
        if m:
            week = date.fromisoformat(m.group(1))
            continue
        m = ITEM_RE.match(line.strip())
        if m and week is not None:
            text = m.group(2).strip()
            raw_urls = URL_RE.findall(text)
            urls = []
            for raw in raw_urls:
                url = clean_url(raw)
                if url is None:
                    dropped_truncated += 1
                else:
                    urls.append(url)
            if tag == "Digest":
                # Title is everything before the first link; the title itself
                # may contain " — " (e.g. a parenthetical aside), so don't
                # split on the first em dash.
                first_url = URL_RE.search(text)
                title = text[: first_url.start()].strip() if first_url else text
                title = re.sub(r"\s*—\s*$", "", title).strip()
            else:
                title = tpu_title(text)
            entries.append(
                {"week": week, "title": title, "urls": urls, "tag": tag}
            )
    return entries, dropped_truncated


def tpu_title(text: str) -> str:
    """Derive a title from a TPU-scan prose item without inventing wording.

    Take the lead clause (up to the first ': ', '; ' or '. '), strip trailing
    metadata parentheticals, and cap the length.
    """
    no_urls = URL_RE.sub("", text)
    no_urls = re.sub(r"\s+", " ", no_urls).strip()
    no_urls = re.sub(r"[\s—–-]+$", "", no_urls).strip()
    # First clause boundary.
    cut = len(no_urls)
    for sep in (": ", "; ", ". "):
        i = no_urls.find(sep)
        if 10 <= i < cut:
            cut = i
    title = no_urls[:cut].strip()
    title = re.sub(r"[\s—–;:,.-]+$", "", title).strip()
    # The cut can strand an opening paren; close it rather than leaving
    # unbalanced punctuation in the title.
    if title.count("(") > title.count(")"):
        title += ")" * (title.count("(") - title.count(")"))
    # Strip trailing metadata parentheticals, e.g. "(Nov 16, CONFIRMED, ...)".
    meta = re.compile(
        r"\(([^)]*(Nov|Oct|Sep|Dec|CONFIRMED|RUMOR|via|vendor|press|analyst)[^)]*)\)\s*$"
    )
    while True:
        m = meta.search(title)
        if not m:
            break
        title = title[: m.start()].strip()
    if len(title) > TITLE_CAP:
        title = title[:TITLE_CAP].rsplit(" ", 1)[0].rstrip() + "…"
    return title


def distinctive_tokens(title: str):
    # Keep dotted versions whole ("v6.1", "0.33.0") so they count as tokens.
    toks = set(re.findall(r"[a-z0-9]+(?:\.[0-9]+)*", title.lower()))
    return {t for t in toks if len(t) >= 3 and t not in STOPWORDS}


def git(*args):
    out = subprocess.run(
        ["git", "-C", str(REPO_ROOT), *args],
        capture_output=True, text=True, check=True,
    )
    return out.stdout


def parse_ingests():
    """Ingest commits on this branch since INGEST_SINCE."""
    entries = []
    log = git(
        "log",
        "--format=%x1f%H%x1f%ad%x1f%s",
        "--date=short",
        f"--since={INGEST_SINCE}",
        "HEAD",
    )
    for line in log.splitlines():
        if not line.strip():
            continue
        _, sha, datestr, subject = line.split("\x1f")
        if not INGEST_SUBJECT.match(subject):
            continue
        numstat = git("show", "--numstat", "--format=", sha)
        touched = []
        for row in numstat.splitlines():
            parts = row.split("\t")
            if len(parts) != 3:
                continue
            added, _deleted, fname = parts
            if not fname.endswith(".md"):
                continue
            if not fname.startswith(SECTION_DIRS):
                continue
            try:
                n = int(added)
            except ValueError:
                continue
            touched.append((fname, n))
        if not touched:
            continue  # ingest that touched no KB section page; skip, don't guess
        if subject.startswith("Ingest") and any(
            f == "references/index.md" for f, _ in touched
        ):
            # Pick/backlog ingests land as reference entries first.
            best_file = "references/index.md"
        else:
            best_file = max(touched, key=lambda t: t[1])[0]
        d = date.fromisoformat(datestr)
        entries.append(
            {
                "week": monday_of(d),
                "title": subject,
                "urls": ["/" + best_file],
                "tag": "Ingested",
            }
        )
    return entries


def dedupe(all_entries):
    """Newest first; digest wins ties; drop URL and same-week title dupes.

    A URL only dedupes across a different tag or a different week — two
    findings in the same week that cite the same source page are distinct.
    Title overlap needs >= 2 shared distinctive tokens, so a one-token
    title (e.g. "vLLM v0.32.0") cannot spuriously match another item.
    """
    priority = {"Digest": 0, "TPU scan": 1, "Ingested": 2}
    ordered = sorted(
        all_entries,
        key=lambda e: (-e["week"].toordinal(), priority[e["tag"]]),
    )
    kept = []
    seen_urls = {}  # norm url -> (week, tag) of the kept entry
    dropped = {"url": 0, "title": 0}
    for e in ordered:
        urls = [u for u in e["urls"] if not u.startswith("/")]
        dupe = False
        for u in urls:
            prev = seen_urls.get(norm_url(u))
            if prev and (prev[0] != e["week"] or prev[1] != e["tag"]):
                dupe = True
        if dupe:
            dropped["url"] += 1
            continue
        # Same-week title overlap against already-kept entries of another tag.
        toks = distinctive_tokens(e["title"])
        overlap = False
        if len(toks) >= 2:
            for k in kept:
                if k["week"] != e["week"] or k["tag"] == e["tag"]:
                    continue
                ktoks = distinctive_tokens(k["title"])
                if (
                    len(toks & ktoks) >= 2
                    and len(toks & ktoks) / len(toks) >= OVERLAP_THRESHOLD
                ):
                    overlap = True
                    break
        if overlap:
            dropped["title"] += 1
            continue
        for u in urls:
            seen_urls[norm_url(u)] = (e["week"], e["tag"])
        kept.append(e)
    return kept, dropped


def render(entries, dropped_truncated, dropped):
    today = datetime.now(USER_TZ).isoformat(timespec="seconds")
    counts = {}
    for e in entries:
        counts[e["tag"]] = counts.get(e["tag"], 0) + 1
    lines = [
        "---",
        "type: Index",
        'title: "Feed"',
        'description: "Time-ordered feed of findings and ingestions"',
        f"timestamp: {today}",
        "---",
        "",
        "# Feed",
        "",
        "Time-ordered feed of findings from the weekly AI performance digest "
        "and the weekly TPU deep scan, plus entries ingested into this "
        "knowledge base. Newest week first; each entry is the ledger's own "
        "title and link — no summaries are invented here.",
        "",
        "Regenerated by `scripts/build_feed.py` from the digest and TPU-scan "
        "ledgers and the repo's git history; re-run the script after a weekly "
        "digest, TPU scan, or merged ingest to refresh it.",
        "",
    ]
    weeks = sorted({e["week"] for e in entries}, reverse=True)
    tag_order = {"Digest": 0, "TPU scan": 1, "Ingested": 2}
    for week in weeks:
        lines.append(f"## Week of {week.isoformat()}")
        lines.append("")
        week_entries = [e for e in entries if e["week"] == week]
        week_entries.sort(key=lambda e: tag_order[e["tag"]])
        for e in week_entries:
            if e["urls"]:
                title_md = f"[{e['title']}]({e['urls'][0]})"
                extra = "".join(
                    f" · [source {i + 2}]({u})"
                    for i, u in enumerate(e["urls"][1:])
                )
            else:
                title_md = e["title"]
                extra = ""
            lines.append(f"- {title_md}{extra} — *{e['tag']}*")
        lines.append("")
    return "\n".join(lines), counts


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--digest-seen", type=Path, default=DIGEST_SEEN)
    ap.add_argument("--tpu-seen", type=Path, default=TPU_SEEN)
    ap.add_argument("--output", type=Path, default=REPO_ROOT / "feed.md")
    args = ap.parse_args()

    digest, drop_d = parse_ledger(args.digest_seen, "Digest")
    tpu, drop_t = parse_ledger(args.tpu_seen, "TPU scan")
    ingests = parse_ingests()
    kept, dropped = dedupe(digest + tpu + ingests)
    text, counts = render(kept, drop_d + drop_t, dropped)
    args.output.write_text(text)
    print(f"Wrote {args.output} ({len(kept)} entries)")
    for tag in ("Digest", "TPU scan", "Ingested"):
        print(f"  {tag}: {counts.get(tag, 0)}")
    print(f"Dropped as duplicates: {dropped}")
    print(f"Dropped truncated ledger URLs: {drop_d + drop_t}")


if __name__ == "__main__":
    main()
