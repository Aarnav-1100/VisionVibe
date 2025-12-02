#!/usr/bin/env python3
import os
import json
import time
import sys
from typing import Any, Dict, List, Optional

# Attempt to import Google GenAI SDK
try:
    import google.genai as genai
    from google.genai.types import GenerateContentConfig
except Exception as e:
    print("Missing dependency: please install the Google GenAI SDK with `pip install -U google-genai`.")
    raise

# Configuration (can be overridden via environment variables)
VISION_VIBE_MODEL = os.environ.get("GEMINI_MODEL", "gemini-2.5-flash")
USE_VERTEX = os.environ.get("GOOGLE_GENAI_USE_VERTEXAI", "false").lower() in ("1", "true", "yes")
GENAI_CLIENT_KWARGS: Dict[str, Any] = {}
if USE_VERTEX:
    project = os.environ.get("GOOGLE_CLOUD_PROJECT")
    location = os.environ.get("GOOGLE_CLOUD_LOCATION")
    if project and location:
        GENAI_CLIENT_KWARGS = {"vertexai_project": project, "vertexai_location": location}
    else:
        print("GOOGLE_GENAI_USE_VERTEXAI is set but GOOGLE_CLOUD_PROJECT or GOOGLE_CLOUD_LOCATION is missing.")


def genai_client() -> genai.Client:
    """Create and return a genai.Client instance."""
    if GENAI_CLIENT_KWARGS:
        return genai.Client(**GENAI_CLIENT_KWARGS)
    return genai.Client()


def call_generate_content(client: genai.Client, model: str, prompt: str, max_retries: int = 3, thinking_budget: int = 0) -> str:
    """Call the model and return plain text. Retries on transient failures.

    thinking_budget is optional and forwarded to the SDK if provided.
    """
    attempt = 0
    while True:
        try:
            cfg = GenerateContentConfig(thinking_budget=thinking_budget) if thinking_budget else None
            resp = client.models.generate_content(model=model, prompt=prompt, config=cfg) if cfg else client.models.generate_content(model=model, prompt=prompt)

            # Try to get a clean text first
            try:
                txt = resp.text()
            except Exception:
                # Fall back to concatenating candidates
                txt = "\n".join([c.content for c in getattr(resp, "candidates", [])])

            return txt.strip()

        except Exception as exc:
            attempt += 1
            if attempt >= max_retries:
                raise
            time.sleep(1 + attempt)


def parse_json_strict(text: str) -> Optional[Any]:
    """Attempt to parse JSON directly; if it fails, try to extract the first JSON object found inside the text."""
    if not text:
        return None
    try:
        return json.loads(text)
    except Exception:
        # find first { and last }
        s = text.find('{')
        e = text.rfind('}')
        if s != -1 and e != -1 and e > s:
            try:
                return json.loads(text[s:e+1])
            except Exception:
                return None
        return None


# Prompt templates (renamed to reflect "Vision Vibe")
PROMPT_GEN_CAREERS = (
    "You are Vision Vibe, an assistant that proposes career or professional paths tailored to a short user profile."
    "\nRespond ONLY with JSON in the following format: {\"careers\": [\"Name (short)\", ...]}."
    "\nUser profile:\nActivities: {activities}\nStrengths: {strengths}\nWork style: {work_style}\n"
    "\nReturn up to {n} candidate career/path names as an array. Do not include any commentary."
)

PROMPT_GEN_QUESTIONS = (
    "You are Vision Vibe. For the following list of candidate career paths: {careers}, generate up to {max_q} multiple-choice questions"
    " that help distinguish between these careers. Respond ONLY with JSON of the form:"
    " {\"questions\": [{\"id\": 1, \"text\": \"...\", \"choices\": [\"Yes\", \"Sometimes\", \"No\"]}, ...]}"
    "Each choice must be implicitly mapped to numeric scores: Yes=1.0, Sometimes=0.5, No=0.0."
)

PROMPT_RANK_AND_EXPLAIN = (
    "You are Vision Vibe. Based on the user's profile, a set of candidate careers, and the user's numeric answers to a set of distinguishing questions,"
    " produce a JSON response of the form:"
    " {\"ranked\": [{\"career\": \"...\", \"score\": 0.0-1.0, \"explanation\": \"one-sentence reason\"}, ...]}"
    " Return at least the top 5 or all candidates if fewer than 5. Scores must be between 0.0 and 1.0. Respond ONLY with JSON."
)


def ask_profile_interactively() -> Dict[str, str]:
    """Collect a short profile from the user via CLI."""
    print("\n--- Vision Vibe: Quick profile (short answers please) ---")
    activities = input("List a few activities you enjoy (comma separated): \n> ").strip()
    strengths = input("What are your top strengths or skills (brief): \n> ").strip()
    work_style = input("Preferred work style? (e.g. remote, team-based, hands-on, flexible): \n> ").strip()
    return {"activities": activities, "strengths": strengths, "work_style": work_style}


def generate_candidate_careers_with_gemini(client: genai.Client, profile: Dict[str, str], n: int = 10) -> List[str]:
    prompt = PROMPT_GEN_CAREERS.format(activities=profile.get("activities", ""), strengths=profile.get("strengths", ""), work_style=profile.get("work_style", ""), n=n)
    raw = call_generate_content(client, VISION_VIBE_MODEL, prompt)

    parsed = parse_json_strict(raw)
    if parsed and isinstance(parsed, dict) and isinstance(parsed.get("careers"), list):
        careers = [c.strip() for c in parsed["careers"] if isinstance(c, str)]
        return careers[:n]

    # fallback: try to extract lines that look like career names
    lines = [line.strip('- *\n ') for line in raw.splitlines() if line.strip()]
    candidates = []
    for line in lines:
        if len(candidates) >= n:
            break
        if len(line.split()) <= 6 and len(line) > 3:
            candidates.append(line)
    if candidates:
        return candidates[:n]

    return ["Software Engineer", "Data Scientist", "Product Designer"]


def generate_questions_for_candidates(client: genai.Client, careers: List[str], max_q: int = 6) -> List[Dict[str, Any]]:
    prompt = PROMPT_GEN_QUESTIONS.format(careers=careers, max_q=max_q)
    raw = call_generate_content(client, VISION_VIBE_MODEL, prompt)
    parsed = parse_json_strict(raw)
    if parsed and isinstance(parsed, dict) and isinstance(parsed.get("questions"), list):
        return parsed["questions"]

    # fallback sample questions
    return [
        {"id": 1, "text": "Do you enjoy working with data and numbers?", "choices": ["Yes", "Sometimes", "No"]},
        {"id": 2, "text": "Do you prefer a stable predictable routine over a changing environment?", "choices": ["Yes", "Sometimes", "No"]},
    ]


def ask_questions_and_collect(questions: List[Dict[str, Any]]) -> Dict[int, Dict[str, Any]]:
    print("\n--- Vision Vibe: Quick questions to refine recommendations ---")
    answers: Dict[int, Dict[str, Any]] = {}
    for q in questions:
        qid = q.get("id") or q.get("idx") or (questions.index(q) + 1)
        text = q.get("text", "")
        choices = q.get("choices", ["Yes", "Sometimes", "No"])[:3]
        print(f"\nQ{qid}: {text}")
        for i, c in enumerate(choices, start=1):
            print(f"  {i}. {c}")
        while True:
            resp = input("Enter 1, 2 or 3: \n> ").strip()
            if resp in ("1", "2", "3"):
                idx = int(resp)
                val = 1.0 if idx == 1 else 0.5 if idx == 2 else 0.0
                answers[int(qid)] = {"choice_index": idx, "value": val}
                break
            else:
                print("Please answer 1, 2, or 3.")
    return answers


def ask_gemini_to_rank(client: genai.Client, careers: List[str], questions: List[Dict[str, Any]], answers: Dict[int, Dict[str, Any]], profile: Dict[str, str]) -> List[Dict[str, Any]]:
    payload = {
        "profile": profile,
        "careers": careers,
        "questions": questions,
        "answers": [{"id": k, "value": v["value"]} for k, v in answers.items()]
    }
    prompt = PROMPT_RANK_AND_EXPLAIN + "\n\nUser payload:\n" + json.dumps(payload, indent=2)
    raw = call_generate_content(client, VISION_VIBE_MODEL, prompt)
    parsed = parse_json_strict(raw)
    if parsed and isinstance(parsed, dict) and isinstance(parsed.get("ranked"), list):
        return parsed["ranked"]

    # fallback heuristic: score by average of answers, bias some careers heuristically
    avg = 0.0
    if answers:
        avg = sum(v["value"] for v in answers.values()) / len(answers)
    ranked = []
    for c in careers:
        score = avg
        if "data" in c.lower():
            score = min(1.0, score + 0.1)
        ranked.append({"career": c, "score": round(score, 3), "explanation": "Auto-generated fallback explanation."})
    ranked.sort(key=lambda x: x["score"], reverse=True)
    return ranked


def main() -> None:
    print("\n=== Vision Vibe (formerly Career Akinator) ===\n")
    client = genai_client()

    profile = ask_profile_interactively()
    candidates = generate_candidate_careers_with_gemini(client, profile, n=10)

    print("\nCandidate paths:")
    for i, c in enumerate(candidates, start=1):
        print(f"  {i}. {c}")

    questions = generate_questions_for_candidates(client, candidates, max_q=6)
    answers = ask_questions_and_collect(questions)

    ranked = ask_gemini_to_rank(client, candidates, questions, answers, profile)

    print("\n--- Final recommendations from Vision Vibe ---")
    for i, entry in enumerate(ranked, start=1):
        print(f"{i}. {entry.get('career')}  (score: {entry.get('score')})")
        print(f"   {entry.get('explanation')}")


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\nInterrupted. Goodbye.")
    except Exception as exc:
        print(f"Error: {exc}")
        raise
