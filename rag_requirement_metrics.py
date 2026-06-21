"""
RAG Requirement Quality Metrics — INCOSE GtWR V4
=================================================
Uses NVIDIA API (LLM-as-Judge) to score each requirement in
LKA SWE 1 Requirement.csv against INCOSE Guide to Writing Requirements V4 rules,
fetching context dynamically from the Qdrant RAG engine.

Output:
  • Console summary table
  • rag_metrics_report.html  — rich visual report
  • rag_metrics_results.csv  — raw scores for further analysis

Usage:
  python rag_requirement_metrics.py
"""

import os
import re
import json
import time
import textwrap
import csv
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime
import pandas as pd

# Add the current directory to path so we can import local modules
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from Model.llm import LLMManager
from RagEngine.rag_engine import RAGEngine

# ─────────────────────────────────────────────────────────────────────────────
# CONFIG
# ─────────────────────────────────────────────────────────────────────────────
# Find the requirements CSV file using fallbacks
csv_fallbacks = [
    "LKA_SWE_1_Requirement.csv",
    "LKA SWE 1 Requirement.csv",
    "artefacts/Lane Keep Assist/LKA SWE 1 Requirement.csv",
    "RequirementValidator/artefacts/Lane Keep Assist/LKA SWE 1 Requirement.csv",
    "../RequirementValidator/artefacts/Lane Keep Assist/LKA SWE 1 Requirement.csv",
    os.path.join(os.path.dirname(os.path.abspath(__file__)), "artefacts", "Lane Keep Assist", "LKA SWE 1 Requirement.csv")
]

CSV_PATH = None
for p in csv_fallbacks:
    if os.path.exists(p):
        CSV_PATH = p
        break

OUTPUT_HTML  = "rag_metrics_report.html"
OUTPUT_CSV   = "rag_metrics_results.csv"
MAX_WORKERS  = 3          # parallel LLM calls (stay conservative with rate limits)
BATCH_SIZE   = 10         # reqs per LLM call (cuts API calls 10×)
RETRY_LIMIT  = 3
RETRY_DELAY  = 5          # seconds base backoff

# ─────────────────────────────────────────────────────────────────────────────
# INCOSE RULES (sourced directly from GtWR V4 Summary Sheet)
# grouped into 10 quality dimensions we can score 0–2 each
# ─────────────────────────────────────────────────────────────────────────────
INCOSE_RULE_GROUPS = {
    "Accuracy (R1–R6)": {
        "rules": [
            "R1 – Structured Statement: Conforms to one agreed pattern (IF/THEN, WHILE, WHEN, WHERE + subject + SHALL + action + condition).",
            "R2 – Active Voice: Responsible entity is the subject of the sentence.",
            "R3 – Appropriate Subject-Verb: Subject and verb are appropriate to the entity.",
            "R4 – Defined Terms: All technical terms are clearly defined or standard.",
            "R5 – Definite Articles: Uses 'the' rather than 'a' for specific references.",
            "R6 – Common Units of Measure: Quantities include explicit, consistent units.",
        ],
        "score_key": "accuracy",
    },
    "Non-Ambiguity (R7–R17)": {
        "rules": [
            "R7 – No Vague Terms: Avoids 'some', 'any', 'approximately', 'adequate', etc.",
            "R8 – No Escape Clauses: Avoids 'where possible', 'if necessary', 'as appropriate', etc.",
            "R9 – No Open-Ended Clauses: Avoids 'including but not limited to', 'etc.'",
            "R12-R14 – Correct Grammar/Spelling/Punctuation.",
            "R15 – Logical Expressions: Uses [X AND Y], [X OR Y] convention for logic.",
            "R16 – No 'Not': Avoids negation where possible.",
            "R17 – No Oblique Symbol: Avoids '/' except in units or fractions.",
        ],
        "score_key": "non_ambiguity",
    },
    "Singularity (R18–R22)": {
        "rules": [
            "R18 – Single Thought Sentence: One capability or constraint per requirement.",
            "R19 – No Combinators: Avoids 'and', 'or', 'then', 'unless', 'but', 'whereas', etc. joining separate obligations.",
            "R20 – No Purpose Phrases: Avoids 'in order to', 'so that', 'to enable'.",
            "R21 – No Parentheses: Avoids brackets containing subordinate clarifications.",
            "R22 – Enumeration: Sets enumerated explicitly, not with group nouns.",
        ],
        "score_key": "singularity",
    },
    "Completeness (R23–R25)": {
        "rules": [
            "R23 – Supporting Reference: Complex behaviours reference a diagram or ICD.",
            "R24 – No Pronouns: No personal or indefinite pronouns (it, they, this, that).",
            "R25 – No Headings: Requirement is self-contained; relies on no heading for meaning.",
        ],
        "score_key": "completeness",
    },
    "Quantification (R33–R35)": {
        "rules": [
            "R33 – Range of Values: Quantities are given with a range or tolerance, not a single absolute.",
            "R34 – Measurable Performance: Specific, measurable targets are provided.",
            "R35 – Temporal Dependencies: Timing is stated explicitly (ms, cycles); no 'eventually', 'before', 'after'.",
        ],
        "score_key": "quantification",
    },
    "Verifiability (C7)": {
        "rules": [
            "C7 – Verifiable: The requirement can be objectively verified/tested.",
            "R34 – Measurable target provided against which test can be executed.",
        ],
        "score_key": "verifiability",
    },
    "Necessity (C1)": {
        "rules": [
            "C1 – Necessary: The requirement defines a real capability, constraint, or quality factor — not an obvious or redundant statement.",
        ],
        "score_key": "necessity",
    },
    "Uniqueness (R29–R30)": {
        "rules": [
            "R29 – Classification: Requirement is correctly classified by the aspect it addresses.",
            "R30 – Unique Expression: Appears once and only once — no duplication of intent with nearby requirements.",
        ],
        "score_key": "uniqueness",
    },
    "Realism (R26, C6)": {
        "rules": [
            "R26 – No Absolutes: Avoids '100%', 'always', 'never', 'all', 'every'.",
            "C6 – Feasible: Requirement can be realised within known technical/cost/schedule constraints.",
        ],
        "score_key": "realism",
    },
    "Conformance (R39, C9)": {
        "rules": [
            "R39 – Style Guide: Follows project-wide style (keyword: WHILE/WHEN/IF/WHERE + THE SYSTEM SHALL).",
            "C9 – Conforming: Conforms to the INCOSE agreed pattern and style guide.",
        ],
        "score_key": "conformance",
    },
}

DIMENSION_KEYS = [v["score_key"] for v in INCOSE_RULE_GROUPS.values()]
DIMENSION_NAMES = list(INCOSE_RULE_GROUPS.keys())

# ─────────────────────────────────────────────────────────────────────────────
# LLM JUDGE PROMPT
# ─────────────────────────────────────────────────────────────────────────────
SYSTEM_PROMPT = """You are an expert Systems Engineer and INCOSE-certified requirements analyst.
You evaluate software/system requirement statements against the INCOSE Guide to Writing Requirements V4 rules.

For each requirement, score EACH of the 10 quality dimensions on a scale:
  2 = Fully compliant
  1 = Partially compliant / minor issues
  0 = Non-compliant / major issues

Also provide:
  - "issues": list of specific rule violations found (e.g. "R19: uses 'and' to join two obligations")
  - "suggestions": list of concrete improvement suggestions

Return ONLY a JSON array (no markdown fences, no commentary) — one object per requirement:
[
  {
    "id": "SYS_REQ_XXXX",
    "accuracy": 0-2,
    "non_ambiguity": 0-2,
    "singularity": 0-2,
    "completeness": 0-2,
    "quantification": 0-2,
    "verifiability": 0-2,
    "necessity": 0-2,
    "uniqueness": 0-2,
    "realism": 0-2,
    "conformance": 0-2,
    "issues": ["..."],
    "suggestions": ["..."]
  },
  ...
]"""

def build_user_prompt(requirements_batch: list[dict], context_block: str) -> str:
    """Build prompt for a batch of requirements, incorporating the RAG context."""
    rules_summary = "\n".join(
        f"\n### {dim_name}\n" + "\n".join(f"  - {r}" for r in cfg["rules"])
        for dim_name, cfg in INCOSE_RULE_GROUPS.items()
    )

    reqs_text = "\n\n".join(
        f"ID: {r['ID']}\nContent: {r['Content']}"
        for r in requirements_batch
    )

    return f"""## INCOSE GtWR V4 — Quality Dimensions to Score

{rules_summary}

---

## Retrieved INCOSE Guideline Context (from RAG vectorDB)

{context_block}

---

## Requirements to Evaluate (batch of {len(requirements_batch)})

{reqs_text}

Evaluate each requirement against all 10 dimensions, leveraging the retrieved INCOSE context to justify your grading, and return the JSON array."""


# ─────────────────────────────────────────────────────────────────────────────
# RAG RETRIEVAL
# ─────────────────────────────────────────────────────────────────────────────
def get_rag_context_for_batch(batch: list[dict], rag: RAGEngine) -> str:
    """Retrieve guidelines context for each requirement in the batch and combine them."""
    combined_guidelines = []
    seen_texts = set()
    
    # Generic query to ensure standard guidelines are always retrieved as base
    generic_query = "INCOSE Guide to Writing Requirements, atomicity, clarity, modal verbs, verifiability, shall"
    try:
        generic_res = rag.search(search_text=generic_query, collection_name="All Collections", top_k=3)
        for r in generic_res:
            text = r["payload"].get("text", "")
            if text and text not in seen_texts:
                seen_texts.add(text)
                combined_guidelines.append(text)
    except Exception as e:
        print(f"  ⚠ Generic RAG search failed: {e}")

    # Specific query for each requirement in the batch
    for req in batch:
        req_content = req.get("Content", "")
        if not req_content:
            continue
        try:
            search_res = rag.search(
                search_text=f"INCOSE rules: {req_content}",
                collection_name="All Collections",
                top_k=2
            )
            for r in search_res:
                text = r["payload"].get("text", "")
                if text and text not in seen_texts:
                    seen_texts.add(text)
                    combined_guidelines.append(text)
        except Exception:
            pass
    # Print the retrieved guidelines context from Qdrant
    print(f"\n[Qdrant RAG Ingestion] Batch: {[r['ID'] for r in batch]}")
    print(f"[Qdrant RAG Ingestion] Retrieved {len(combined_guidelines)} unique context guidelines:")
    for idx, text in enumerate(combined_guidelines, 1):
        truncated_text = text[:120].replace('\n', ' ') + '...' if len(text) > 120 else text
        print(f"  {idx}. {truncated_text}")
    print("-" * 60)

    # Format the guidelines
    context_texts = [f"Guideline Rule #{idx}:\n{text}" for idx, text in enumerate(combined_guidelines, 1)]
    return "\n\n".join(context_texts) if context_texts else "No context rules found in RAG."


# ─────────────────────────────────────────────────────────────────────────────
# API CALL WITH RETRY
# ─────────────────────────────────────────────────────────────────────────────
def call_llm_judge(batch: list[dict], context_block: str, llm: LLMManager) -> list[dict]:
    """Call NVIDIA LLM API via LLMManager. Returns list of scored dicts."""
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": build_user_prompt(batch, context_block)}
    ]
    
    for attempt in range(RETRY_LIMIT):
        try:
            response = llm.get_response(messages, stream=False)
            raw = response.choices[0].message.content.strip()

            # Strip markdown fences if model wraps anyway
            raw = re.sub(r"^```(?:json)?\s*", "", raw)
            raw = re.sub(r"\s*```$", "", raw)

            results = json.loads(raw)
            if isinstance(results, list):
                return results
            raise ValueError("Expected JSON array")

        except (json.JSONDecodeError, ValueError) as e:
            if attempt == RETRY_LIMIT - 1:
                print(f"  ⚠ JSON parse failed after {RETRY_LIMIT} attempts: {e}")
                # Return zeroed fallback
                return [
                    {
                        "id": r["ID"],
                        **{k: 0 for k in DIMENSION_KEYS},
                        "issues": ["LLM failed to parse response"],
                        "suggestions": [],
                    }
                    for r in batch
                ]
            time.sleep(RETRY_DELAY)

        except Exception as e:
            print(f"  ✗ API error: {e}")
            wait = RETRY_DELAY * (2 ** attempt)
            print(f"  ⏳ Waiting {wait}s before retry...")
            time.sleep(wait)

    return []


# ─────────────────────────────────────────────────────────────────────────────
# MAIN PIPELINE
# ─────────────────────────────────────────────────────────────────────────────
def run_metrics(csv_path: str, rag: RAGEngine, llm: LLMManager) -> pd.DataFrame:
    df = pd.read_csv(csv_path)
    # Take only the first 50 requirements to analyze
    df = df.head(50)
    requirements = df[["ID", "Content"]].to_dict("records")

    # Split into batches
    batches = [requirements[i:i + BATCH_SIZE] for i in range(0, len(requirements), BATCH_SIZE)]
    total_batches = len(batches)

    print(f"\n{'='*60}")
    print(f"  INCOSE RAG Requirement Quality Metrics (NVIDIA NIM)")
    print(f"  Requirements: {len(requirements)} | Batches: {total_batches} (size {BATCH_SIZE})")
    print(f"  Model: {llm.model_name}")
    print(f"{'='*60}\n")

    all_results: list[dict] = []

    def process_batch(batch_idx: int, batch: list[dict]):
        print(f"  → Processing batch {batch_idx+1}/{total_batches} ({len(batch)} reqs)...")
        context_block = get_rag_context_for_batch(batch, rag)
        results = call_llm_judge(batch, context_block, llm)
        return results

    with ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
        futures = {
            executor.submit(process_batch, i, batch): i
            for i, batch in enumerate(batches)
        }
        for future in as_completed(futures):
            batch_results = future.result()
            if batch_results:
                all_results.extend(batch_results)

    # Build results dataframe
    rows = []
    req_map = {r["ID"]: r for r in requirements}

    for res in all_results:
        req_id = res.get("id", "UNKNOWN")
        original = req_map.get(req_id, {})
        row = {
            "ID": req_id,
            "Content": original.get("Content", ""),
        }
        for key in DIMENSION_KEYS:
            row[key] = int(res.get(key, 0))
        row["total_score"] = sum(row[key] for key in DIMENSION_KEYS)
        row["max_score"]   = len(DIMENSION_KEYS) * 2
        row["pct_score"]   = round(100 * row["total_score"] / row["max_score"], 1)
        row["issues"]      = "; ".join(res.get("issues", []))
        row["suggestions"] = "; ".join(res.get("suggestions", []))
        rows.append(row)

    result_df = pd.DataFrame(rows)
    # Sort by original order
    id_order = [r["ID"] for r in requirements]
    result_df["_order"] = result_df["ID"].map({v: i for i, v in enumerate(id_order)})
    result_df = result_df.sort_values("_order").drop(columns=["_order"]).reset_index(drop=True)

    return result_df


# ─────────────────────────────────────────────────────────────────────────────
# HTML REPORT GENERATOR
# ─────────────────────────────────────────────────────────────────────────────
def score_color(pct: float) -> str:
    if pct >= 80:   return "#22c55e"   # green
    if pct >= 60:   return "#f59e0b"   # amber
    return "#ef4444"                    # red


def score_bg(pct: float) -> str:
    if pct >= 80:   return "rgba(34,197,94,0.12)"
    if pct >= 60:   return "rgba(245,158,11,0.12)"
    return "rgba(239,68,68,0.12)"


def dim_cell_color(score: int) -> str:
    if score == 2:  return "#bbf7d0"   # light green
    if score == 1:  return "#fef08a"   # light yellow
    return "#fecaca"                    # light red


def generate_html_report(df: pd.DataFrame, output_path: str, model_name: str):
    now = datetime.now().strftime("%Y-%m-%d %H:%M")

    # Aggregate stats
    total = len(df)
    avg_pct = df["pct_score"].mean()
    pass_80  = (df["pct_score"] >= 80).sum()
    pass_60  = ((df["pct_score"] >= 60) & (df["pct_score"] < 80)).sum()
    fail     = (df["pct_score"] < 60).sum()

    # Per-dimension averages
    dim_avgs = {key: df[key].mean() for key in DIMENSION_KEYS}
    worst_dim = min(dim_avgs, key=dim_avgs.get)
    best_dim  = max(dim_avgs, key=dim_avgs.get)

    # Dimension average bars HTML
    dim_bars_html = ""
    for (dim_name, cfg), key in zip(INCOSE_RULE_GROUPS.items(), DIMENSION_KEYS):
        avg = dim_avgs[key]
        pct = avg / 2 * 100
        color = score_color(pct)
        dim_bars_html += f"""
        <div class="dim-bar-row">
          <div class="dim-label">{dim_name}</div>
          <div class="bar-track">
            <div class="bar-fill" style="width:{pct:.0f}%;background:{color}"></div>
          </div>
          <div class="dim-val" style="color:{color}">{avg:.2f}/2</div>
        </div>"""

    # Dimension header cells
    dim_headers = "".join(
        '<th title="{}">{}</th>'.format(
            INCOSE_RULE_GROUPS[dn]["rules"][0],
            key.replace("_", " ").title()
        )
        for dn, key in zip(DIMENSION_NAMES, DIMENSION_KEYS)
    )

    # Table rows
    table_rows = ""
    for _, row in df.iterrows():
        pct = row["pct_score"]
        badge_color = score_color(pct)
        badge_bg    = score_bg(pct)
        grade = "PASS" if pct >= 80 else ("WARN" if pct >= 60 else "FAIL")

        dim_cells = "".join(
            f'<td class="dim-cell" style="background:{dim_cell_color(int(row[k]))}">{int(row[k])}</td>'
            for k in DIMENSION_KEYS
        )

        issues_html = ""
        if row["issues"]:
            items = row["issues"].split("; ")
            issues_html = "<ul class='issue-list'>" + "".join(f"<li>{i}</li>" for i in items if i) + "</ul>"

        content_short = textwrap.shorten(str(row["Content"]), width=120, placeholder="…")

        table_rows += f"""
        <tr>
          <td class="req-id">{row['ID']}</td>
          <td class="req-content" title="{row['Content']}">{content_short}</td>
          {dim_cells}
          <td>
            <span class="badge" style="color:{badge_color};background:{badge_bg};border:1px solid {badge_color}">
              {pct:.0f}% {grade}
            </span>
          </td>
          <td class="issues-cell">{issues_html}</td>
        </tr>"""

    # Bottom issues summary — top 10 most common
    all_issues = []
    for iss_str in df["issues"].dropna():
        for iss in iss_str.split("; "):
            if iss.strip():
                all_issues.append(iss.strip())

    from collections import Counter
    issue_counts = Counter(all_issues).most_common(10)
    top_issues_html = "<ol class='top-issues'>" + "".join(
        f"<li><span class='count-badge'>{cnt}</span> {iss}</li>"
        for iss, cnt in issue_counts
    ) + "</ol>"

    html = f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>INCOSE RAG Requirement Metrics — LKA SWE-1</title>
<style>
  :root {{
    --bg: #0f172a; --surface: #1e293b; --border: #334155;
    --text: #e2e8f0; --muted: #94a3b8; --accent: #6366f1;
  }}
  * {{ box-sizing: border-box; margin: 0; padding: 0; }}
  body {{ font-family: 'Segoe UI', system-ui, sans-serif; background: var(--bg); color: var(--text); padding: 24px; }}
  h1 {{ font-size: 1.8rem; font-weight: 700; background: linear-gradient(135deg,#6366f1,#a78bfa,#ec4899);
        -webkit-background-clip: text; -webkit-text-fill-color: transparent; margin-bottom: 4px; }}
  .subtitle {{ color: var(--muted); font-size: 0.9rem; margin-bottom: 24px; }}

  /* Summary cards */
  .cards {{ display: grid; grid-template-columns: repeat(auto-fit,minmax(160px,1fr)); gap: 16px; margin-bottom: 28px; }}
  .card {{ background: var(--surface); border: 1px solid var(--border); border-radius: 12px; padding: 18px; text-align:center; }}
  .card .val {{ font-size: 2rem; font-weight: 700; }}
  .card .lbl {{ color: var(--muted); font-size: 0.78rem; margin-top: 4px; text-transform: uppercase; letter-spacing: .05em; }}

  /* Dimension bars */
  .dim-section {{ background: var(--surface); border: 1px solid var(--border); border-radius: 12px; padding: 20px; margin-bottom: 24px; }}
  .dim-section h2 {{ font-size: 1rem; font-weight: 600; margin-bottom: 16px; color: #a78bfa; }}
  .dim-bar-row {{ display: flex; align-items: center; gap: 12px; margin-bottom: 10px; }}
  .dim-label {{ width: 200px; font-size: 0.8rem; color: var(--muted); flex-shrink: 0; }}
  .bar-track {{ flex: 1; height: 10px; background: var(--border); border-radius: 5px; overflow: hidden; }}
  .bar-fill  {{ height: 100%; border-radius: 5px; transition: width .3s; }}
  .dim-val   {{ width: 60px; text-align: right; font-size: 0.82rem; font-weight: 600; }}

  /* Top issues */
  .issues-section {{ background: var(--surface); border: 1px solid var(--border); border-radius: 12px; padding: 20px; margin-bottom: 24px; }}
  .issues-section h2 {{ font-size: 1rem; font-weight: 600; margin-bottom: 14px; color: #f87171; }}
  .top-issues {{ padding-left: 20px; }}
  .top-issues li {{ font-size: 0.82rem; color: var(--text); margin-bottom: 6px; line-height: 1.5; }}
  .count-badge {{ background: rgba(239,68,68,.2); color:#f87171; border-radius:4px; padding:1px 6px;
                  font-size:0.72rem; font-weight:700; margin-right:6px; }}

  /* Table */
  .table-section {{ background: var(--surface); border: 1px solid var(--border); border-radius: 12px; padding: 20px; overflow-x: auto; }}
  .table-section h2 {{ font-size: 1rem; font-weight: 600; margin-bottom: 14px; color: #60a5fa; }}
  table {{ width: 100%; border-collapse: collapse; font-size: 0.78rem; }}
  th {{ background: #0f172a; color: var(--muted); text-transform: uppercase; font-size: 0.68rem;
        letter-spacing:.04em; padding: 8px 6px; border-bottom: 1px solid var(--border); white-space: nowrap;
        cursor: help; }}
  td {{ padding: 7px 6px; border-bottom: 1px solid rgba(51,65,85,.5); vertical-align: top; }}
  tr:hover td {{ background: rgba(99,102,241,.06); }}
  .req-id {{ font-weight: 600; color: #a78bfa; white-space: nowrap; }}
  .req-content {{ color: var(--muted); max-width: 260px; line-height: 1.4; }}
  .dim-cell {{ text-align: center; font-weight: 700; font-size: 0.82rem; border-radius: 4px; }}
  .badge {{ padding: 3px 8px; border-radius: 20px; font-size: 0.72rem; font-weight: 700; white-space: nowrap; }}
  .issue-list {{ padding-left: 14px; color: #fca5a5; font-size: 0.74rem; }}
  .issue-list li {{ margin-bottom: 3px; line-height: 1.4; }}
  .issues-cell {{ max-width: 280px; }}

  .footer {{ text-align:center; color: var(--muted); font-size: 0.75rem; margin-top: 28px; }}
</style>
</head>
<body>

<h1>🔍 INCOSE Requirement Quality Metrics</h1>
<p class="subtitle">LKA SWE-1 Requirements · INCOSE GtWR V4 · Generated {now} · Model: {model_name}</p>

<!-- Summary Cards -->
<div class="cards">
  <div class="card">
    <div class="val">{total}</div>
    <div class="lbl">Total Requirements</div>
  </div>
  <div class="card">
    <div class="val" style="color:{score_color(avg_pct)}">{avg_pct:.1f}%</div>
    <div class="lbl">Average Score</div>
  </div>
  <div class="card">
    <div class="val" style="color:#22c55e">{pass_80}</div>
    <div class="lbl">✅ PASS (≥80%)</div>
  </div>
  <div class="card">
    <div class="val" style="color:#f59e0b">{pass_60}</div>
    <div class="lbl">⚠️ WARN (60–79%)</div>
  </div>
  <div class="card">
    <div class="val" style="color:#ef4444">{fail}</div>
    <div class="lbl">❌ FAIL (&lt;60%)</div>
  </div>
  <div class="card">
    <div class="val" style="color:#a78bfa;font-size:1rem">{best_dim.replace('_',' ').title()}</div>
    <div class="lbl">🏆 Best Dimension</div>
  </div>
  <div class="card">
    <div class="val" style="color:#f87171;font-size:1rem">{worst_dim.replace('_',' ').title()}</div>
    <div class="lbl">⚠️ Weakest Dimension</div>
  </div>
</div>

<!-- Dimension Bars -->
<div class="dim-section">
  <h2>📊 Average Score by INCOSE Quality Dimension (out of 2.0)</h2>
  {dim_bars_html}
</div>

<!-- Top Issues -->
<div class="issues-section">
  <h2>🚩 Top 10 Most Common Rule Violations</h2>
  {top_issues_html}
</div>

<!-- Detailed Table -->
<div class="table-section">
  <h2>📋 Requirement-Level Detail</h2>
  <p style="color:var(--muted);font-size:0.78rem;margin-bottom:12px">
    Hover column headers to see which INCOSE rule they represent. Cell colours: 🟢 2=Pass · 🟡 1=Partial · 🔴 0=Fail
  </p>
  <table>
    <thead>
      <tr>
        <th>ID</th>
        <th>Content (truncated)</th>
        {dim_headers}
        <th>Score</th>
        <th>Issues Found</th>
      </tr>
    </thead>
    <tbody>
      {table_rows}
    </tbody>
  </table>
</div>

<p class="footer">INCOSE GtWR V4 Summary Sheet — INCOSE-TP-2010-006-04 | June 2023 · LLM-as-Judge via NVIDIA NIM {model_name}</p>
</body>
</html>"""

    with open(output_path, "w", encoding="utf-8") as f:
        f.write(html)
    print(f"\n  ✅ HTML report saved → {output_path}")


# ─────────────────────────────────────────────────────────────────────────────
# CONSOLE SUMMARY
# ─────────────────────────────────────────────────────────────────────────────
def print_console_summary(df: pd.DataFrame):
    print(f"\n{'='*60}")
    print("  RESULTS SUMMARY")
    print(f"{'='*60}")
    print(f"  Total requirements :  {len(df)}")
    print(f"  Average score      :  {df['pct_score'].mean():.1f}%")
    print(f"  PASS (≥80%)        :  {(df['pct_score']>=80).sum()}")
    print(f"  WARN (60–79%)      :  {((df['pct_score']>=60)&(df['pct_score']<80)).sum()}")
    print(f"  FAIL (<60%)        :  {(df['pct_score']<60).sum()}")
    print(f"\n  Dimension Averages (out of 2.0):")
    for key, dim_name in zip(DIMENSION_KEYS, DIMENSION_NAMES):
        avg = df[key].mean()
        bar = "█" * int(avg * 10) + "░" * (20 - int(avg * 10))
        print(f"    {dim_name:<32}  {bar}  {avg:.2f}")
    print(f"\n  Bottom 5 requirements by score:")
    worst = df.nsmallest(5, "pct_score")[["ID", "pct_score", "issues"]]
    for _, row in worst.iterrows():
        issues_short = textwrap.shorten(str(row["issues"]), 70, placeholder="…")
        print(f"    {row['ID']}  {row['pct_score']:.0f}%  {issues_short}")
    print(f"\n{'='*60}")


# ─────────────────────────────────────────────────────────────────────────────
# ENTRY POINT
# ─────────────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    # Check for API key (loaded from env or secrets internally)
    llm = LLMManager()
    if not llm.client.api_key:
        print("ERROR: NVIDIA_API_KEY environment variable is not set.")
        print("  Please make sure NVIDIA_API_KEY or API_KEY is set in your .env or api_key.env.")
        exit(1)

    if not CSV_PATH:
        print("ERROR: Requirements CSV file not found in any of the expected locations.")
        exit(1)

    print(f"[*] Loading RAG engine...")
    rag = RAGEngine(llm_manager=llm)
    # Sync documents from Qdrant
    rag.load_trained_engine()

    # Run evaluation
    result_df = run_metrics(CSV_PATH, rag, llm)

    # Save CSV
    result_df.to_csv(OUTPUT_CSV, index=False)
    print(f"  ✅ Raw results saved → {OUTPUT_CSV}")

    # Generate HTML report
    generate_html_report(result_df, OUTPUT_HTML, llm.model_name)

    # Print console summary
    print_console_summary(result_df)
