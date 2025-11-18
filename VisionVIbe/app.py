# app.py
"""
Streamlit Career Akinator (Gemini-backed)

Features:
- Uses Google GenAI (Gemini) via google-genai SDK (reads GEMINI_API_KEY from environment)
- Robust parsing of LLM responses (fenced markdown removal, fallback extraction)
- Threaded LLM calls with timeout & retries to avoid UI freeze
- Simple file-based caching to reduce API calls
- Streamlit UI: profile collection -> LLM generates careers+questions -> user answers -> LLM ranks

Before running:
  pip install streamlit google-genai
  # set GEMINI_API_KEY in the same terminal:
  # PowerShell:
  $Env:GEMINI_API_KEY="YOUR_KEY_HERE"
  streamlit run app.py
"""

import os
import json
import re
import time
import hashlib
import traceback
import threading
from typing import Any, Dict, List, Optional, Tuple
import streamlit as st

# Try import GenAI client
try:
    from google import genai
    from google.genai import types as genai_types
except Exception as e:
    st.error("Missing 'google-genai' package. Install with: pip install google-genai")
    raise

# ---------------------------
# Configuration
# ---------------------------
DEFAULT_MODEL = os.getenv("GEMINI_MODEL", "models/gemini-2.5-flash")
CACHE_DIR = os.getenv("CACHE_DIR", ".cache_llm")
LLM_TIMEOUT = int(os.getenv("LLM_TIMEOUT", "25"))  # seconds
LLM_RETRIES = int(os.getenv("LLM_RETRIES", "1"))   # retries on failure
# Create cache dir if not exists
os.makedirs(CACHE_DIR, exist_ok=True)

# ---------------------------
# Helpers: robust response extraction & JSON cleaning
# ---------------------------
def extract_response_text(resp) -> str:
    """
    Robustly extract text from various shapes of SDK responses.
    """
    if resp is None:
        return ""

    # If resp has attribute 'text' (callable or property)
    if hasattr(resp, "text"):
        maybe = getattr(resp, "text")
        try:
            if callable(maybe):
                return maybe()
            else:
                return str(maybe)
        except Exception:
            # fallback to string conversion
            try:
                return str(maybe)
            except Exception:
                pass

    # If resp has 'candidates' (older/newer SDK shapes)
    if hasattr(resp, "candidates"):
        try:
            parts = []
            for c in resp.candidates:
                content = getattr(c, "content", None)
                if content and getattr(content, "parts", None):
                    for p in content.parts:
                        parts.append(getattr(p, "text", str(p)))
                else:
                    parts.append(str(c))
            if parts:
                return "\n".join(parts)
        except Exception:
            pass

    # final fallback: try stringifying the whole object
    try:
        return str(resp)
    except Exception:
        return ""

def clean_markdown_json(text: str) -> str:
    """
    Strip common markdown fencing and whitespace around JSON-like text.
    """
    if not isinstance(text, str):
        text = str(text)
    s = text.strip()

    # Remove triple-backtick fences optionally with language tag
    # e.g. ```json\n{...}\n``` or ```\n[...]\n```
    fence_match = re.match(r"^```(?:\w+)?\s*(.*)\s*```$", s, flags=re.DOTALL | re.IGNORECASE)
    if fence_match:
        return fence_match.group(1).strip()

    # Remove leading descriptive lines until the first { or [
    # Also remove single-line fences or surrounding backticks
    s = re.sub(r"^```[^\n]*\n", "", s)
    s = re.sub(r"\n```$", "", s)
    if s.startswith("`") and s.endswith("`"):
        s = s[1:-1].strip()

    # Find first JSON start
    idx_candidates = [i for i in (s.find("{"), s.find("[")) if i >= 0]
    if idx_candidates:
        idx = min(idx_candidates)
        s = s[idx:].strip()

    return s

def extract_json_from_text(text: str) -> Any:
    """
    Try to parse JSON from an LLM response. Raises ValueError with helpful info if it fails.
    """
    cleaned = clean_markdown_json(text)

    # Direct try
    try:
        return json.loads(cleaned)
    except Exception:
        pass

    # Fallback: find balanced JSON substrings (object or array)
    candidates = []

    def find_balanced(s: str, open_ch: str, close_ch: str):
        starts = [m.start() for m in re.finditer(re.escape(open_ch), s)]
        for s_index in starts:
            depth = 0
            for i in range(s_index, len(s)):
                if s[i] == open_ch:
                    depth += 1
                elif s[i] == close_ch:
                    depth -= 1
                    if depth == 0:
                        yield s[s_index : i + 1]

    for cand in find_balanced(cleaned, "{", "}"):
        candidates.append(cand)
    for cand in find_balanced(cleaned, "[", "]"):
        candidates.append(cand)

    # Try candidates (prefer longer)
    candidates = sorted(set(candidates), key=len, reverse=True)
    last_exc = None
    for cand in candidates:
        try:
            return json.loads(cand)
        except Exception as e:
            last_exc = e
            continue

    raise ValueError(
        "Failed to parse JSON from model output.\n\n"
        f"Cleaned snippet (truncated):\n{cleaned[:2000]!r}\n\n"
        f"Original raw (truncated):\n{text[:2000]!r}\n\n"
        f"Last JSON error: {last_exc}"
    )

# ---------------------------
# Helpers: caching
# ---------------------------
def cache_key_from_profile(profile: Dict[str,str]) -> str:
    """
    Derive a stable cache key from the profile dictionary.
    """
    s = json.dumps(profile, sort_keys=True, ensure_ascii=False)
    h = hashlib.sha256(s.encode("utf-8")).hexdigest()
    return h

def load_cache(key: str) -> Optional[Dict[str, Any]]:
    path = os.path.join(CACHE_DIR, f"{key}.json")
    if os.path.exists(path):
        try:
            with open(path, "r", encoding="utf-8") as f:
                return json.load(f)
        except Exception:
            return None
    return None

def save_cache(key: str, value: Dict[str, Any]) -> None:
    path = os.path.join(CACHE_DIR, f"{key}.json")
    try:
        with open(path, "w", encoding="utf-8") as f:
            json.dump(value, f, ensure_ascii=False, indent=2)
    except Exception:
        pass

# ---------------------------
# GenAI client wrapper & safe call runner
# ---------------------------
def get_gemini_client():
    """Create genai.Client with detection of GEMINI_API_KEY or Vertex mode."""
    # If user explicitly set key in env, use it
    api_key = os.getenv("GEMINI_API_KEY")
    use_vertex = os.getenv("GOOGLE_GENAI_USE_VERTEXAI", "false").lower() in ("1", "true", "yes")
    if api_key:
        # pass api_key explicit to be safe in some SDK setups
        return genai.Client(api_key=api_key)
    if use_vertex:
        project = os.getenv("GOOGLE_CLOUD_PROJECT")
        location = os.getenv("GOOGLE_CLOUD_LOCATION", "us-central1")
        if not project:
            raise RuntimeError("Vertex mode requested but GOOGLE_CLOUD_PROJECT is not set.")
        return genai.Client(project=project, location=location, vertexai=True)
    # if nothing set, genai.Client() will try to find creds, but raise helpful error
    return genai.Client()

def call_genai_generate(client, model: str, prompt: str, thinking_budget: int = 0) -> str:
    """
    Call GenAI and return the textual result. This is a single-call wrapper.
    """
    # Build config if available
    try:
        config = genai_types.GenerateContentConfig(
            thinking_config = genai_types.ThinkingConfig(thinking_budget=thinking_budget)
        )
        resp = client.models.generate_content(model=model, contents=prompt, config=config)
    except TypeError:
        # Some sdk versions accept different args
        resp = client.models.generate_content(model=model, contents=prompt)
    # Extract text robustly
    return extract_response_text(resp)

def run_llm_with_timeout(fn, args=(), kwargs=None, timeout=LLM_TIMEOUT, max_retries=LLM_RETRIES) -> Dict[str, Any]:
    """
    Run a blocking function in a background thread with timeout and retries.
    Returns dict with keys: ok(bool), result(str) or error(str), attempts(int)
    """
    kwargs = kwargs or {}
    attempt = 0
    last_error = None
    while attempt <= max_retries:
        attempt += 1
        result_holder = {"result": None, "error": None}

        def target():
            try:
                result_holder["result"] = fn(*args, **kwargs)
            except Exception as e:
                result_holder["error"] = traceback.format_exc()

        thread = threading.Thread(target=target, daemon=True)
        thread.start()
        thread.join(timeout=timeout)
        if thread.is_alive():
            # timed out
            last_error = f"Timeout after {timeout}s on attempt {attempt}"
            # thread stays daemon -> will exit on program end; we continue retrying
        else:
            if result_holder["error"]:
                last_error = result_holder["error"]
            else:
                return {"ok": True, "result": result_holder["result"], "attempts": attempt}
        time.sleep(1 + attempt * 0.5)
    return {"ok": False, "error": last_error, "attempts": attempt}

# ---------------------------
# Prompt templates (simple & conservative)
# ---------------------------
PROMPT_GENERATE_CAREERS_AND_QUESTIONS = """
You are a helpful career-suggestion assistant.
Given the user's short profile, return JSON ONLY in the following format (no extra text):

{{
  "careers": ["Career 1", "Career 2", "..."],
  "questions": [
    {{
      "id":"q1",
      "text":"Short question text",
      "choices": [
        {{"label":"Yes-style choice", "value": 1.0}},
        {{"label":"Sometimes choice", "value": 0.5}},
        {{"label":"No-style choice", "value": 0.0}}
      ]
    }},
    ...
  ]
}}

Rules:
- Return up to {max_careers} careers.
- Return up to {max_questions} discriminative questions (each with exactly 3 choices).
- Values must be numeric 1.0 / 0.5 / 0.0.
- Output must be valid JSON only. No markdown fences, no commentary.
- Keep text concise.
"""

PROMPT_RANK_AND_EXPLAIN = """
You are a career-ranking assistant.
Given candidate careers, the questions (text), and the user's numeric responses (1.0,0.5,0.0),
return JSON ONLY in this format:

{{
  "ranking": [
    {{"career":"Career A", "score": 0.87}},
    ...
  ],
  "explanations": [
    {{"career":"Career A", "explanation": "one-sentence reason why it's a fit"}},
    ...
  ]
}}

Rules:
- Scores: floats between 0.0 and 1.0.
- Provide a ranking for all candidate careers (higher is better).
- Provide explanations for top 3 only.
- Return valid JSON only. No other commentary.
"""

# ---------------------------
# Streamlit UI
# ---------------------------
st.set_page_config(page_title="Career Akinator (Gemini)", layout="centered")
st.title("Career Akinator — AI-assisted career suggestions")

# show whether GEMINI key is visible (safe debug)
with st.sidebar:
    st.header("Status & Settings")
    st.write("Model:", DEFAULT_MODEL)
    st.write("LLM timeout (s):", LLM_TIMEOUT)
    st.write("Retries:", LLM_RETRIES)
    st.write("Cache dir:", CACHE_DIR)
    # show presence of API key without printing it
    key_present = bool(os.getenv("GEMINI_API_KEY"))
    st.write("GEMINI key present?", "✅" if key_present else "❌ (set GEMINI_API_KEY)")

st.info("Fill the small profile below and press **Generate**. The AI will suggest careers and a few questions.")

with st.form(key="profile_form"):
    col1, col2 = st.columns([2, 1])
    with col1:
        activities = st.text_input("What activities or subjects do you enjoy? (e.g., coding, drawing, helping others)")
        strengths = st.text_input("What are your strengths or skills? (e.g., math, communication, creativity)")
    with col2:
        work_style = st.selectbox("Preferred work style", ["team", "solo", "hands-on", "desk", "flexible", "stable hours"])
        max_careers = st.slider("Max careers", 3, 12, value=8)
        max_questions = st.slider("Max questions", 2, 8, value=5)
    submit = st.form_submit_button("Generate")

if submit:
    profile = {"activities": activities or "", "strengths": strengths or "", "work_style": work_style or ""}
    cache_key = cache_key_from_profile(profile)
    cached = load_cache(cache_key)
    if cached:
        st.success("Loaded suggestions from cache (to save API calls).")
        careers = cached.get("careers", [])
        questions = cached.get("questions", [])
    else:
        # build prompt
        prompt = PROMPT_GENERATE_CAREERS_AND_QUESTIONS.format(max_careers=max_careers, max_questions=max_questions)
        prompt += "\n\nUser profile:\n"
        prompt += f"- activities: {profile['activities']}\n"
        prompt += f"- strengths: {profile['strengths']}\n"
        prompt += f"- work_style: {profile['work_style']}\n"

        st.info("Calling the AI to generate candidate careers and questions...")
        client = None
        try:
            client = get_gemini_client()
        except Exception as e:
            st.error("Failed to create Gemini client. See message below.")
            st.code(str(e))
            st.stop()

        # run LLM safely
        with st.spinner("Generating candidates & questions (this may take a few seconds)..."):
            call = lambda: call_genai_generate(client, DEFAULT_MODEL, prompt)
            out = run_llm_with_timeout(call, timeout=LLM_TIMEOUT, max_retries=LLM_RETRIES)
        if not out.get("ok"):
            st.error("AI call failed or timed out.")
            st.code(out.get("error"))
            st.stop()

        raw = out["result"]
        st.text("Raw AI output (truncated):")
        st.code(raw[:2000])

        try:
            parsed = extract_json_from_text(raw)
            # basic validation and normalization
            careers = parsed.get("careers", []) if isinstance(parsed, dict) else []
            questions = parsed.get("questions", []) if isinstance(parsed, dict) else []
            # ensure choices numeric & count 3
            norm_questions = []
            for i, q in enumerate(questions):
                qid = q.get("id") or f"q{i+1}"
                text = q.get("text") or q.get("question") or ""
                choices = q.get("choices") or q.get("options") or []
                # ensure three choices
                if len(choices) < 3:
                    # fallback default
                    choices = [
                        {"label": "Yes", "value": 1.0},
                        {"label": "Sometimes", "value": 0.5},
                        {"label": "No", "value": 0.0},
                    ]
                # coerce numeric values
                for ch in choices[:3]:
                    try:
                        ch["value"] = float(ch.get("value", 0.0))
                    except Exception:
                        ch["value"] = 0.0
                    ch["label"] = str(ch.get("label", ""))
                norm_questions.append({"id": qid, "text": text, "choices": choices[:3]})
            questions = norm_questions
            if not careers:
                st.warning("AI returned no careers. Using a small fallback list.")
                careers = ["Software Engineer", "Data Scientist", "Teacher"]
            # save cache
            save_cache(cache_key, {"careers": careers, "questions": questions, "profile": profile})
            st.success("Generated and cached careers & questions.")
        except Exception as e:
            st.error("Failed to parse AI output as JSON.")
            st.code(str(e))
            st.warning("Shown raw output above for debugging. Try again or adjust the profile.")
            st.stop()

    # Present careers and questions to user
    st.markdown("### Candidate careers")
    for i, c in enumerate(careers, start=1):
        st.write(f"{i}. {c}")

    st.markdown("---")
    st.markdown("### Questions")
    response_map = {}
    with st.form(key="qa_form"):
        for idx, q in enumerate(questions, start=1):
            st.write(f"**Q{idx}.** {q['text']}")
            opts = [ch["label"] for ch in q["choices"]]
            choice = st.radio(f"Select (Q{idx})", opts, key=f"q_{idx}")
            # map selected label back to numeric
            sel_index = opts.index(choice)
            numeric = q["choices"][sel_index]["value"]
            response_map[q["id"]] = {"choice_index": sel_index + 1, "value": numeric}
        submitted_answers = st.form_submit_button("Submit Answers")

    if submitted_answers:
        st.info("Sending answers to AI to produce final ranking...")
        client = None
        try:
            client = get_gemini_client()
        except Exception as e:
            st.error("Failed to create Gemini client. See message below.")
            st.code(str(e))
            st.stop()

        # prepare rank prompt
        answers_list = [{"id": qid, "question": next((qq["text"] for qq in questions if qq["id"]==qid), ""), "selected_value": resp["value"]} for qid, resp in response_map.items()]
        prompt_rank = PROMPT_RANK_AND_EXPLAIN + "\n\nCandidate careers:\n"
        prompt_rank += "\n".join(f"- {c}" for c in careers)
        prompt_rank += "\n\nUser answers:\n" + json.dumps(answers_list, indent=2)

        with st.spinner("AI is ranking careers..."):
            call = lambda: call_genai_generate(client, DEFAULT_MODEL, prompt_rank)
            out = run_llm_with_timeout(call, timeout=LLM_TIMEOUT, max_retries=LLM_RETRIES)

        if not out.get("ok"):
            st.error("AI ranking call failed or timed out.")
            st.code(out.get("error"))
            st.stop()

        raw_rank = out["result"]
        st.text("Raw ranking output (truncated):")
        st.code(raw_rank[:2000])

        try:
            parsed_rank = extract_json_from_text(raw_rank)
            ranking = parsed_rank.get("ranking", []) if isinstance(parsed_rank, dict) else []
            explanations = parsed_rank.get("explanations", []) if isinstance(parsed_rank, dict) else []
            if not ranking:
                st.warning("AI did not return ranking — using fallback scoring.")
                # fallback simple heuristic: keep original order with equal score
                ranking = [{"career": c, "score": 0.5} for c in careers]
            st.success("Final ranking:")
            for i, item in enumerate(ranking, start=1):
                career = item.get("career", "Unknown")
                score = float(item.get("score", 0.0))
                st.write(f"{i}. **{career}** — confidence {score:.2f}")
            if explanations:
                st.markdown("---")
                st.markdown("### Explanations (top picks)")
                for ex in explanations[:3]:
                    st.write(f"- **{ex.get('career')}**: {ex.get('explanation')}")
        except Exception as e:
            st.error("Failed to parse ranking JSON from AI.")
            st.code(str(e))
            st.warning("Shown raw ranking output above for debugging.")

# Footer
st.markdown("---")
st.markdown("Choose your career wisely! Powered by Google Gemini and Streamlit.")

