#!/usr/bin/env python3
"""Generate the Top Deep Learning Projects list from the GitHub Search API.

Collects repositories matching a set of deep-learning-related topics and
free-text searches, deduplicates them, filters out curated lists, and writes
the result as a markdown table (optionally as the full README.md).

The GitHub Search API caps every query at 1000 results, so each query is
sliced into star ranges (geometric bisection) until every slice fits.

Auth: uses $GITHUB_TOKEN / $GH_TOKEN, or falls back to `gh auth token`.
With a token the search rate limit is 30 requests/min; without, 10/min.

Usage:
    python3 scripts/generate_stats.py --readme README.md \
        --cache /tmp/search_cache.json --dump /tmp/repos.json
"""

import argparse
import datetime
import json
import math
import os
import subprocess
import sys
import time
import urllib.error
import urllib.parse
import urllib.request

API_URL = "https://api.github.com/search/repositories"
MAX_RESULTS_PER_QUERY = 1000  # hard cap of the GitHub Search API
MAX_STARS = 600_000           # above the most-starred repo on GitHub
PER_PAGE = 100

# Topic pages to crawl (exact `topic:` matches).
TOPICS = [
    # classics
    "deep-learning", "machine-learning", "neural-network",
    "tensorflow", "pytorch", "jax",
    "computer-vision", "nlp", "natural-language-processing",
    "reinforcement-learning", "speech-recognition",
    # classic subfields
    "object-detection", "image-segmentation", "semantic-segmentation",
    "image-classification", "image-generation", "face-recognition",
    "gan", "generative-adversarial-network", "ocr", "machine-translation",
    "text-to-speech", "speech-to-text", "recommender-system",
    "graph-neural-networks", "pose-estimation", "yolo", "keras", "onnx",
    "neural-networks", "deeplearning", "embeddings", "anomaly-detection",
    # post-2020 landscape
    "llm", "large-language-models", "transformers", "generative-ai",
    "gpt", "chatgpt", "llama", "stable-diffusion", "diffusion-models",
    "text-to-image", "video-generation", "rag", "ai-agents", "agentic-ai",
    "multimodal", "fine-tuning", "rlhf", "llm-inference", "huggingface",
    "openai", "mlops", "artificial-intelligence",
    # LLM ecosystem
    "text-generation", "llmops", "llm-serving", "llm-agents",
    "llm-training", "llm-evaluation", "quantization", "qlora",
    "mcp", "model-context-protocol", "autonomous-agents",
    "multi-agent-systems", "conversational-ai", "ai-assistant", "chatbot",
    "prompt-engineering", "vector-database", "vector-search",
    "langchain", "ollama", "whisper",
    "mistral", "qwen", "deepseek", "claude", "anthropic", "gemini",
]

# Free-text searches over name / description / topics.
SEARCHES = [
    "deep learning", "machine learning", "neural network",
    "computer vision", "reinforcement learning", "speech recognition",
    "object detection", "image segmentation", "image generation",
    "video generation", "face recognition", "text to speech",
    "pose estimation", "graph neural network", "vision transformer",
    "vision language model", "foundation model", "world model",
    "gaussian splatting", "inference", "trained model",
    "tensorflow", "pytorch", "transformer",
    "llm", "large language model", "generative ai", "gpt", "llama",
    "stable diffusion", "diffusion model", "ai agent",
    # quoted phrases are not stemmed by the search API: add plurals/variants
    "language model", "language models", "large language models",
    "diffusion models", "generative model", "generative models",
    "image synthesis", "gradient boosting", "text embeddings", "tts",
    "face swap", "voice cloning", "voice conversion",
    # LLM ecosystem
    "mcp server", "model context protocol", "prompt engineering",
    "retrieval augmented", "vector database", "ai assistant",
    "coding agent", "code assistant", "chatbot",
    "deepseek", "qwen", "mistral", "claude",
    "agents", "multi-agent",
]

# Major projects that no generic keyword reaches (no topics, terse
# descriptions). Fetched directly and merged into the results.
SEEDS = [
    "triton-lang/triton",
    "ml-explore/mlx",
    "NVIDIA/NeMo",
    "lllyasviel/Fooocus",
    "google-deepmind/deepmind-research",
    "suno-ai/bark",
    "oobabooga/text-generation-webui",
]

# Repos whose description marks them as curated lists rather than projects.
SKIP_PATTERNS = ["curated list", "ranked list", "list of"]

# Never include these repos (this list itself, star-farmed/name-squatting repos).
EXCLUDE = {
    "aymericdamien/TopDeepLearning",
    "multica-ai/andrej-karpathy-skills",  # single CLAUDE.md file, not a project
}

_session = {"token": None, "last_request": 0.0, "sleep": 2.1, "requests": 0}


def get_token():
    token = os.environ.get("GITHUB_TOKEN") or os.environ.get("GH_TOKEN")
    if not token:
        try:
            token = subprocess.run(
                ["gh", "auth", "token"], capture_output=True, text=True, timeout=10
            ).stdout.strip() or None
        except (OSError, subprocess.SubprocessError):
            token = None
    return token


def api_get(query, page):
    """One search request. Returns the parsed response, or None when GitHub
    replies 422 past the 1000-result window (treated as end of pagination)."""
    params = urllib.parse.urlencode(
        {"q": query, "per_page": PER_PAGE, "page": page, "sort": "stars", "order": "desc"}
    )
    headers = {
        "Accept": "application/vnd.github+json",
        "X-GitHub-Api-Version": "2022-11-28",
        "User-Agent": "TopDeepLearning-stats",
    }
    if _session["token"]:
        headers["Authorization"] = "Bearer " + _session["token"]

    for attempt in range(10):
        wait = _session["last_request"] + _session["sleep"] - time.time()
        if wait > 0:
            time.sleep(wait)
        _session["last_request"] = time.time()
        _session["requests"] += 1
        req = urllib.request.Request(API_URL + "?" + params, headers=headers)
        try:
            with urllib.request.urlopen(req, timeout=30) as resp:
                return json.load(resp)
        except urllib.error.HTTPError as e:
            if e.code == 422 and page > 1:
                return None
            if e.code in (403, 429):
                retry_after = e.headers.get("Retry-After")
                reset = e.headers.get("X-RateLimit-Reset")
                if retry_after:
                    delay = int(retry_after)
                elif reset:
                    delay = max(int(reset) - time.time(), 5)
                else:
                    delay = 60
                delay = min(delay, 300) + 2
                print("  rate limited, sleeping %.0fs" % delay, flush=True)
                time.sleep(delay)
            elif e.code >= 500:
                time.sleep(10)
            else:
                raise
        except (urllib.error.URLError, TimeoutError, json.JSONDecodeError):
            time.sleep(10)
    raise RuntimeError("giving up on query %r page %d" % (query, page))


def fetch_repo(full_name):
    """Fetch a single repo directly (for SEEDS). Returns None on failure."""
    headers = {
        "Accept": "application/vnd.github+json",
        "X-GitHub-Api-Version": "2022-11-28",
        "User-Agent": "TopDeepLearning-stats",
    }
    if _session["token"]:
        headers["Authorization"] = "Bearer " + _session["token"]
    req = urllib.request.Request("https://api.github.com/repos/" + full_name, headers=headers)
    for _ in range(3):
        try:
            with urllib.request.urlopen(req, timeout=30) as resp:
                return pare(json.load(resp))
        except (urllib.error.URLError, TimeoutError, json.JSONDecodeError):
            time.sleep(5)
    print("warning: could not fetch seed repo %s" % full_name, flush=True)
    return None


def pare(item):
    return {
        "name": item["full_name"],
        "url": item["html_url"],
        "desc": item.get("description") or "",
        "stars": item["stargazers_count"],
        "archived": item.get("archived", False),
        "created": item.get("created_at", ""),
    }


def run_query(query, min_stars):
    """All repos matching `query` with >= min_stars stars, slicing the star
    range until each slice fits in the API's 1000-result window."""
    results = []

    def fetch_range(lo, hi):
        qual = "stars:>=%d" % lo if hi >= MAX_STARS else "stars:%d..%d" % (lo, hi)
        full_q = "%s %s" % (query, qual)
        data = api_get(full_q, 1)
        total = data["total_count"]
        if total > MAX_RESULTS_PER_QUERY and lo < hi:
            # geometric midpoint: star counts are log-distributed
            mid = min(max(int(math.sqrt(lo * hi)), lo), hi - 1)
            fetch_range(lo, mid)
            fetch_range(mid + 1, hi)
            return
        results.extend(pare(i) for i in data["items"])
        for page in range(2, min(math.ceil(total / PER_PAGE), 10) + 1):
            data = api_get(full_q, page)
            if data is None or not data["items"]:
                break
            results.extend(pare(i) for i in data["items"])

    fetch_range(min_stars, MAX_STARS)
    return results


def keep(repo, min_stars):
    if repo["stars"] < min_stars or repo["name"] in EXCLUDE:
        return False
    desc = repo["desc"].lower()
    name = repo["name"].split("/")[-1].lower()
    if "awesome" in name:
        return False
    return not any(p in desc for p in SKIP_PATTERNS)


def fmt_stars(n):
    if n < 1000:
        return str(n)
    if n < 100_000:
        return ("%.1f" % (n / 1000)).rstrip("0").rstrip(".") + "k"
    return "%dk" % round(n / 1000)


DESC_MAX = 100


def clean_desc(desc):
    desc = " ".join(desc.split())  # collapse whitespace/newlines
    if len(desc) > DESC_MAX:
        desc = desc[: DESC_MAX - 1].rstrip().rstrip("\\") + "…"
    return desc.replace("|", "\\|")


def build_md(repos):
    lines = ["| Project Name | Stars | Description |", "| ------- | ------ | ------ |"]
    for r in repos:
        name = r["name"].split("/")[-1]
        lines.append("|[%s](%s)|%s|%s|" % (name, r["url"], fmt_stars(r["stars"]), clean_desc(r["desc"])))
    return "\n".join(lines) + "\n"


def build_readme(repos):
    header = (
        "# Top Deep Learning Projects\n"
        "A list of popular github projects related to deep learning (ranked by stars).\n"
        "\n"
        "Last Update: %s\n" % datetime.date.today().strftime("%Y.%m.%d")
    )
    return header + build_md(repos)


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    # 5000 keeps the README under GitHub's ~512KB render limit (1000 stars
    # matched ~7300 repos / ~1MB as of 2026).
    ap.add_argument("--min-stars", type=int, default=5000, help="star threshold for the final list")
    ap.add_argument("--max-repos", type=int, default=0, help="cap the list at N repos (0 = no cap)")
    ap.add_argument("--readme", help="write the full README to this path")
    ap.add_argument("--out-md", help="write the bare markdown table to this path")
    ap.add_argument("--dump", help="write all collected repos as JSON to this path")
    ap.add_argument("--cache", help="JSON file caching per-query results (resume support)")
    ap.add_argument("--queries", help="comma-separated raw queries overriding the default sets")
    args = ap.parse_args()

    _session["token"] = get_token()
    if not _session["token"]:
        _session["sleep"] = 6.5  # unauthenticated: 10 requests/min
        print("warning: no GitHub token found, running unauthenticated (slow)", flush=True)

    if args.queries:
        queries = [q.strip() for q in args.queries.split(",") if q.strip()]
    else:
        queries = ["topic:%s" % t for t in TOPICS] + [
            ('"%s"' % s if " " in s else s) + " in:name,description,topics" for s in SEARCHES
        ]

    cache = {}
    if args.cache and os.path.exists(args.cache):
        with open(args.cache) as f:
            cache = json.load(f)

    repos = {}
    for i, q in enumerate(queries, 1):
        if q in cache:
            items = cache[q]
            status = "cached"
        else:
            items = run_query(q, args.min_stars)
            status = "fetched"
            if args.cache:
                cache[q] = items
                with open(args.cache, "w") as f:
                    json.dump(cache, f)
        for r in items:
            prev = repos.get(r["url"])
            if prev is None or r["stars"] > prev["stars"]:
                repos[r["url"]] = r
        print(
            "[%d/%d] %s: %d repos (%s) | unique so far: %d | api requests: %d"
            % (i, len(queries), q, len(items), status, len(repos), _session["requests"]),
            flush=True,
        )

    if not args.queries:
        for seed in SEEDS:
            r = fetch_repo(seed)
            if r and r["url"] not in repos:
                repos[r["url"]] = r

    result = sorted(
        (r for r in repos.values() if keep(r, args.min_stars)),
        key=lambda r: (-r["stars"], r["name"].lower()),
    )
    if args.max_repos:
        result = result[: args.max_repos]

    print("%d repos after filtering (of %d unique collected)" % (len(result), len(repos)), flush=True)

    if args.dump:
        with open(args.dump, "w") as f:
            json.dump(sorted(repos.values(), key=lambda r: -r["stars"]), f, indent=1)
    if args.out_md:
        with open(args.out_md, "w") as f:
            f.write(build_md(result))
    if args.readme:
        with open(args.readme, "w") as f:
            f.write(build_readme(result))


if __name__ == "__main__":
    main()
