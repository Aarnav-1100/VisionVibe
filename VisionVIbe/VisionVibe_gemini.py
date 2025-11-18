#!/usr/bin/env python3
"""
career_akinator_gemini.py

Dynamic career Akinator using Google Gemini (GenAI SDK).

Requires:
  pip install -U google-genai
Auth:
  - Set GEMINI_API_KEY in env (recommended), OR
  - Use Google ADC (gcloud auth application-default login) + set GOOGLE_CLOUD_PROJECT & GOOGLE_CLOUD_LOCATION
"""

import os
import json
import time
import sys
from typing import List, Dict, Any

# Import the Gen AI client per Google docs
try:
    # official style shown in docs: from google import genai
    from google import genai
    from google.genai import types as genai_types
except Exception as e:
    print("Missing google-genai package. Install with: pip install -U google-genai")
    raise

# ---- Configuration ----
# Choose model alias — pick one you have access to (gemini-2.5-flash is common)
GEMINI_MODEL = os.getenv("GEMINI_MODEL", "gemini-2.5-flash")
# Optionally set vertex usage via env var; if you prefer Vertex, set GOOGLE_GENAI_USE_VERTEXAI=true
USE_VERTEX = os.getenv("GOOGLE_GENAI_USE_VERTEXAI", "false").lower() in ("1","true","yes")

# If you want to pass API key directly in code (NOT recommended for production),
# set GEMINI_API_KEY env var or uncomment and set here (quick test only).
# genai.configure(api_key="YOUR_API_KEY_HERE")

# If using Vertex in Google Cloud, you can initialize with project/location
GENAI_CLIENT_KWARGS = {}
if USE_VERTEX:
    # If you want to route requests to Vertex, supply project/location from env
    project = os.getenv("GOOGLE_CLOUD_PROJECT")
    location = os.getenv("GOOGLE_CLOUD_LOCATION", "us-central1")
    if not project:
        print("Warning: GOOGLE_GENAI_USE_VERTEXAI=true but GOOGLE_CLOUD_PROJECT not set.")
    GENAI_CLIENT_KWARGS.update({"project": project, "location": location})


# ---- Small helpers for robust LLM JSON output & retries ----
def genai_client():
    """Create and return a genai.Client() instance (picks up GEMINI_API_KEY or ADC)."""
    # The client constructor uses environment config (GEMINI_API_KEY or ADC) by default.
    # You can also pass api_key=... here if you want to hardcode (not recommended).
    client = genai.Client(**GENAI_CLIENT_KWARGS) if GENAI_CLIENT_KWARGS else genai.Client()
    return client

def call_generate_content(client, model: str, prompt: str, max_retries=3, thinking_budget: int = 0) -> str:
    """Calls the Gemini generate_content API and returns the plain text response (concatenated)."""
    for attempt in range(1, max_retries + 1):
        try:
            # Use the higher-level generate_content API (Python SDK)
            # Accepts simple string contents; config tuned to disable heavy "thinking" by default
            config = genai_types.GenerateContentConfig(
                thinking_config = genai_types.ThinkingConfig(thinking_budget=thinking_budget)
            )
            resp = client.models.generate_content(model=model, contents=prompt, config=config)
            # The SDK response exposes .text() or .candidates; docs show response.text or response.text()
            # We'll try the documented .text() helper, but fallback to raw candidate parsing.
            try:
                text = resp.text()
            except Exception:
                # fallback parsing
                candidates = getattr(resp, "candidates", None)
                if candidates:
                    parts = []
                    for c in candidates:
                        content = getattr(c, "content", None)
                        if content and getattr(content, "parts", None):
                            for p in content.parts:
                                parts.append(getattr(p, "text", str(p)))
                    text = "\n".join(parts)
                else:
                    text = str(resp)
            return text.strip()
        except Exception as e:
            print(f"[GenAI] call error (attempt {attempt}): {e}")
            if attempt == max_retries:
                raise
            time.sleep(1 + attempt * 2)
    raise RuntimeError("Unreachable")

def parse_json_strict(text: str):
    """Try to extract JSON object from an assistant reply robustly."""
    text = text.strip()
    # Quick exact parse
    try:
        return json.loads(text)
    except Exception:
        # Find first and last braces
        start = text.find("{")
        end = text.rfind("}")
        if start != -1 and end != -1 and end > start:
            candidate = text[start:end+1]
            try:
                return json.loads(candidate)
            except Exception:
                pass
    return None

# ---- Prompt templates used with Gemini ----
PROMPT_GEN_CAREERS = """
You are a helpful career-suggestion assistant.
Given a short user profile, return a JSON object exactly in this format:

{{
  "careers": ["Career 1", "Career 2", "..."]
}}

- Provide up to {n} distinct, realistic career suggestions that match the user's profile.
- Do NOT include extra explanation or commentary outside the JSON object.

User profile:
- activities: {activities}
- strengths: {strengths}
- work_style: {work_style}
"""

PROMPT_GEN_QUESTIONS = """
You are a question generator for career diagnosis.
Given this list of candidate careers, produce up to {max_q} multiple-choice questions that help distinguish between these careers.

Return JSON exactly in this format:

{{
  "questions": [
    {{
      "id": "q1",
      "text": "Short question text",
      "choices": [
        {{"label":"Yes-style choice", "value": 1.0}},
        {{"label":"Sometimes-style choice", "value": 0.5}},
        {{"label":"No-style choice", "value": 0.0}}
      ]
    }},
    ...
  ]
}}

Rules:
- Each question must have exactly 3 choices with numeric mapping 1.0 / 0.5 / 0.0.
- Questions should be concise & discriminative for the given careers.
- Do NOT include commentary outside the JSON.

Candidate careers:
{careers_list}
"""

PROMPT_RANK_AND_EXPLAIN = """
You are a career-ranking assistant.
Given candidate careers, the questions (text), and the user's numeric responses (1.0/0.5/0.0), return a JSON object exactly like:

{{
  "ranking": [
    {{"career": "Career A", "score": 0.87}},
    ...
  ],
  "explanations": [
    {{"career": "Career A", "explanation": "one-sentence explanation for why it's a match"}},
    ...
  ]
}}

Rules:
- Scores should be floats between 0.0 and 1.0.
- Provide ranking for all candidate careers (higher score = better match).
- Provide explanations for the top 3 careers only.
- Return only JSON, no extra text.

Candidate careers:
{careers_list}

Questions and user's numeric answers:
{answers_json}
"""

# ---- Main conversational flow ----
def ask_profile_interactively() -> Dict[str,str]:
    print("Welcome — a few quick questions to understand you. Short answers are fine.")
    activities = input("1) What activities or subjects do you enjoy? (e.g., coding, drawing, helping others) > ").strip()
    strengths = input("2) What are your strengths or skills? (e.g., math, communication, creativity) > ").strip()
    work_style = input("3) What work style do you prefer? (e.g., team / solo, hands-on / desk, stable hours) > ").strip()
    return {"activities": activities, "strengths": strengths, "work_style": work_style}

def generate_candidate_careers_with_gemini(client, profile: Dict[str,str], n=10) -> List[str]:
    prompt = PROMPT_GEN_CAREERS.format(
        n=n,
        activities=profile["activities"],
        strengths=profile["strengths"],
        work_style=profile["work_style"],
    )
    raw = call_generate_content(client, GEMINI_MODEL, prompt, thinking_budget=0)
    parsed = parse_json_strict(raw)
    if parsed and isinstance(parsed.get("careers"), list):
        return parsed["careers"][:n]
    # fallback: try splitting lines
    lines = [l.strip("-• ") for l in raw.splitlines() if l.strip()]
    # take short plausible lines
    candidates = []
    for l in lines:
        if len(candidates) >= n:
            break
        if 3 <= len(l) <= 60 and any(c.isalpha() for c in l):
            candidates.append(l)
    return candidates or ["Software Engineer", "Data Scientist", "Teacher"]

def generate_questions_for_candidates(client, careers: List[str], max_q=6) -> List[Dict[str,Any]]:
    prompt = PROMPT_GEN_QUESTIONS.format(careers_list="\n".join(f"- {c}" for c in careers), max_q=max_q)
    raw = call_generate_content(client, GEMINI_MODEL, prompt, thinking_budget=0)
    parsed = parse_json_strict(raw)
    if parsed and isinstance(parsed.get("questions"), list):
        return parsed["questions"][:max_q]
    # fallback simple questions
    fallback = [
        {"id":"q1","text":"Do you enjoy technical problem-solving?","choices":[{"label":"Yes","value":1.0},{"label":"Sometimes","value":0.5},{"label":"No","value":0.0}]},
        {"id":"q2","text":"Do you prefer creative/artistic tasks?","choices":[{"label":"Yes","value":1.0},{"label":"Sometimes","value":0.5},{"label":"No","value":0.0}]},
    ]
    return fallback

def ask_questions_and_collect(questions: List[Dict[str,Any]]) -> Dict[str,Dict[str,Any]]:
    print("\nAnswer the following questions. Choose 1 / 2 / 3 corresponding to the option (or type yes/sometimes/no).")
    answers = {}
    for idx, q in enumerate(questions, start=1):
        qid = q.get("id", f"q{idx}")
        print(f"\nQ{idx}. {q.get('text')}")
        choices = q.get("choices", [])
        for i, ch in enumerate(choices, start=1):
            print(f"  {i}. {ch.get('label')}")
        # collect
        while True:
            ans = input("Your choice (1/2/3) > ").strip().lower()
            if ans in ("1","2","3") and len(choices) >= int(ans):
                sel = int(ans)-1
                value = float(choices[sel].get("value", 0.0))
                answers[qid] = {"choice_index": sel+1, "value": value}
                break
            if ans in ("yes","y","1","true"):
                answers[qid] = {"choice_index": 1, "value": 1.0}
                break
            if ans in ("sometimes","maybe","s","0.5","half"):
                answers[qid] = {"choice_index": 2, "value": 0.5}
                break
            if ans in ("no","n","0","false"):
                answers[qid] = {"choice_index": 3, "value": 0.0}
                break
            print("Please answer 1, 2 or 3 (or yes / sometimes / no).")
    return answers

def ask_gemini_to_rank(client, careers: List[str], questions: List[Dict[str,Any]], answers: Dict[str,Any], profile: Dict[str,str]):
    # Prepare the answer list for prompt
    answers_list = []
    for q in questions:
        qid = q.get("id")
        answers_list.append({
            "id": qid,
            "question": q.get("text"),
            "selected_value": answers.get(qid, {}).get("value")
        })
    prompt = PROMPT_RANK_AND_EXPLAIN.format(
        careers_list="\n".join(f"- {c}" for c in careers),
        answers_json=json.dumps(answers_list, indent=2)
    )
    raw = call_generate_content(client, GEMINI_MODEL, prompt, thinking_budget=0)
    parsed = parse_json_strict(raw)
    if parsed and isinstance(parsed.get("ranking"), list):
        return parsed
    # fallback heuristic ranking
    avg = sum([v["value"] for v in answers.values()]) / max(1, len(answers))
    fallback = {"ranking":[{"career":c,"score":min(1.0, avg + (0.05 if c.lower().find('data')!=-1 else 0.0))} for c in careers],
                "explanations":[{"career":careers[0],"explanation":"Fallback: approximate match based on answers."}]}
    fallback["ranking"] = sorted(fallback["ranking"], key=lambda x:-x["score"])
    return fallback

def main():
    client = genai_client()
    profile = ask_profile_interactively()
    print("\n[1/3] Generating candidate careers from your profile...")
    candidates = generate_candidate_careers_with_gemini(client, profile, n=10)
    print("Candidates:")
    for i,c in enumerate(candidates,1):
        print(f"  {i}. {c}")

    print("\n[2/3] Creating discriminative questions...")
    questions = generate_questions_for_candidates(client, candidates, max_q=6)
    # Normalize questions to expected format
    norm_qs = []
    for i,q in enumerate(questions, start=1):
        qid = q.get("id") or f"q{i}"
        text = q.get("text") or q.get("question") or "..."
        choices = q.get("choices") or q.get("options") or []
        # ensure three choices exist
        if len(choices) < 3:
            choices = choices + [{"label":"Yes","value":1.0},{"label":"Sometimes","value":0.5},{"label":"No","value":0.0}]
        # coerce numeric
        for ch in choices[:3]:
            try:
                ch["value"] = float(ch.get("value", 0.0))
            except Exception:
                ch["value"] = 0.0
            ch["label"] = str(ch.get("label",""))
        norm_qs.append({"id": qid, "text": text, "choices": choices[:3]})

    answers = ask_questions_and_collect(norm_qs)

    print("\n[3/3] Ranking careers and explaining top picks...")
    result = ask_gemini_to_rank(client, candidates, norm_qs, answers, profile)

    print("\n--- Final ranking ---")
    for i,item in enumerate(result.get("ranking", []), start=1):
        career = item.get("career")
        score = float(item.get("score", 0.0))
        print(f"{i}. {career} — {score:.2f}")

    print("\nTop explanations:")
    for ex in result.get("explanations", [])[:3]:
        print(f"- {ex.get('career')}: {ex.get('explanation')}")

if __name__ == "__main__":
    main()