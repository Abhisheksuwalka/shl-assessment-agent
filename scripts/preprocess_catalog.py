"""
preprocess_catalog.py — Step 2.1: Catalog Preprocessing

Loads the raw SHL product catalog, cleans and enriches each entry,
builds a `text_for_embedding` field, and saves as `catalog_processed.json`.

Run once:
    python -m scripts.preprocess_catalog
"""

import json
import re
import sys
from pathlib import Path
from collections import Counter

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
PROJECT_ROOT = Path(__file__).resolve().parent.parent
RAW_CATALOG = PROJECT_ROOT / "data" / "shl_product_catalog.json"
PROCESSED_CATALOG = PROJECT_ROOT / "data" / "catalog_processed.json"

# ---------------------------------------------------------------------------
# Type code mapping (SHL key → single-letter code)
# ---------------------------------------------------------------------------
KEY_TO_CODE = {
    "Knowledge & Skills": "K",
    "Personality & Behavior": "P",
    "Ability & Aptitude": "A",
    "Biodata & Situational Judgment": "B",
    "Competencies": "C",
    "Assessment Exercises": "E",
    "Simulations": "S",
    "Development & 360": "D",
}


def parse_duration(raw: str) -> int | None:
    """
    Parse duration string like '30 minutes' → 30.
    Returns None for empty/unparseable values.
    """
    if not raw or not raw.strip():
        return None
    match = re.search(r"(\d+)", raw.strip())
    return int(match.group(1)) if match else None


def map_keys_to_codes(keys: list[str]) -> list[str]:
    """Map full SHL key names to single-letter codes."""
    codes = []
    for key in keys:
        code = KEY_TO_CODE.get(key)
        if code:
            codes.append(code)
        else:
            print(f"  WARNING: Unknown key '{key}' — skipping")
    return codes


def parse_remote_adaptive(val: str) -> bool:
    """Convert 'yes'/'no' strings to boolean."""
    return val.strip().lower() == "yes" if val else False


def build_text_for_embedding(item: dict) -> str:
    """
    Build a rich text representation for semantic embedding.
    Concatenates name, description, categories, job levels, duration, languages.
    """
    parts = [item["name"]]

    if item.get("description"):
        parts.append(item["description"])

    # Categories (full names for better semantic matching)
    if item.get("keys"):
        parts.append(f"Categories: {', '.join(item['keys'])}")

    # Type codes for keyword matching
    if item.get("type_codes"):
        parts.append(f"Type codes: {', '.join(item['type_codes'])}")

    # Job levels
    if item.get("job_levels"):
        parts.append(f"Job levels: {', '.join(item['job_levels'])}")

    # Duration
    if item.get("duration_minutes") is not None:
        parts.append(f"Duration: {item['duration_minutes']} minutes")

    # Languages
    if item.get("languages"):
        parts.append(f"Languages: {', '.join(item['languages'][:10])}")

    # Adaptive
    if item.get("adaptive"):
        parts.append("Adaptive: yes")

    return ". ".join(parts)


def preprocess_catalog() -> list[dict]:
    """Load, clean, enrich, and return the processed catalog."""

    print(f"Loading raw catalog from: {RAW_CATALOG}")
    with open(RAW_CATALOG, "r", encoding="utf-8") as f:
        raw_data = json.load(f)
    print(f"  Loaded {len(raw_data)} items")

    processed = []
    unknown_keys = Counter()

    for item in raw_data:
        # Parse and enrich fields
        duration_minutes = parse_duration(item.get("duration", ""))
        type_codes = map_keys_to_codes(item.get("keys", []))
        remote = parse_remote_adaptive(item.get("remote", ""))
        adaptive = parse_remote_adaptive(item.get("adaptive", ""))

        enriched = {
            "entity_id": item["entity_id"],
            "name": item["name"],
            "link": item["link"],
            "description": item.get("description", ""),
            "keys": item.get("keys", []),
            "type_codes": type_codes,
            "job_levels": item.get("job_levels", []),
            "languages": item.get("languages", []),
            "duration_minutes": duration_minutes,
            "duration_raw": item.get("duration", ""),
            "remote": remote,
            "adaptive": adaptive,
        }

        # Build embedding text
        enriched["text_for_embedding"] = build_text_for_embedding(enriched)
        processed.append(enriched)

    return processed


def validate_catalog(data: list[dict]) -> bool:
    """Run validation checks and print summary statistics."""
    print("\n===== VALIDATION =====")
    passed = True

    # 1. No null names or links
    null_names = sum(1 for d in data if not d.get("name"))
    null_links = sum(1 for d in data if not d.get("link"))
    print(f"  Null names: {null_names}")
    print(f"  Null links: {null_links}")
    if null_names > 0 or null_links > 0:
        print("  ❌ FAIL: Found null names or links")
        passed = False
    else:
        print("  ✅ No null names or links")

    # 2. All keys mapped
    all_codes = []
    unmapped_items = 0
    for d in data:
        if d["keys"] and not d["type_codes"]:
            unmapped_items += 1
        all_codes.extend(d["type_codes"])
    print(f"\n  Items with keys but no type codes: {unmapped_items}")
    if unmapped_items > 0:
        print("  ❌ FAIL: Some keys did not map")
        passed = False
    else:
        print("  ✅ All keys mapped to codes")

    # 3. Summary statistics
    print("\n--- Type Code Distribution ---")
    for code, count in Counter(all_codes).most_common():
        label = {v: k for k, v in KEY_TO_CODE.items()}.get(code, "?")
        print(f"  {code} ({label}): {count}")

    durations = [d["duration_minutes"] for d in data if d["duration_minutes"] is not None]
    if durations:
        print(f"\n--- Duration (minutes) ---")
        print(f"  Count: {len(durations)} / {len(data)} items have duration")
        print(f"  Min: {min(durations)}, Max: {max(durations)}, Mean: {sum(durations)/len(durations):.1f}")

    lang_counts = Counter()
    for d in data:
        lang_counts.update(d["languages"])
    print(f"\n--- Languages ---")
    print(f"  Items with languages: {sum(1 for d in data if d['languages'])} / {len(data)}")
    print(f"  Unique languages: {len(lang_counts)}")
    print(f"  Top 5: {lang_counts.most_common(5)}")

    jl_counts = Counter()
    for d in data:
        jl_counts.update(d["job_levels"])
    print(f"\n--- Job Levels ---")
    for jl, count in jl_counts.most_common():
        print(f"  {jl}: {count}")

    adaptive_count = sum(1 for d in data if d["adaptive"])
    print(f"\n--- Adaptive: {adaptive_count} / {len(data)} ---")

    # 4. text_for_embedding spot check
    print(f"\n--- Embedding Text Spot Check ---")
    sample = data[0]
    print(f"  Item: {sample['name']}")
    print(f"  Text length: {len(sample['text_for_embedding'])} chars")
    print(f"  Preview: {sample['text_for_embedding'][:200]}...")

    if passed:
        print("\n✅ ALL VALIDATION CHECKS PASSED")
    else:
        print("\n❌ SOME VALIDATION CHECKS FAILED")

    return passed


def main():
    processed = preprocess_catalog()

    # Save
    print(f"\nSaving processed catalog to: {PROCESSED_CATALOG}")
    with open(PROCESSED_CATALOG, "w", encoding="utf-8") as f:
        json.dump(processed, f, indent=2, ensure_ascii=False)
    print(f"  Saved {len(processed)} items")

    # Validate
    valid = validate_catalog(processed)
    if not valid:
        sys.exit(1)


if __name__ == "__main__":
    main()
