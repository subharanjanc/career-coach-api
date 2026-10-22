"""Local checks. `python test_api.py` is free; `--live` runs a real interview."""

import json
import sys
from pathlib import Path

# Embeddable Python's python314._pth keeps the script's folder off sys.path.
sys.path.insert(0, str(Path(__file__).resolve().parent))

from fastapi.testclient import TestClient            # noqa: E402

from api.index import app                            # noqa: E402

client = TestClient(app)
passed = failed = 0


def check(name, ok, detail=""):
    global passed, failed
    if ok:
        passed += 1
        print("  PASS  %s" % name)
    else:
        failed += 1
        print("  FAIL  %s  %s" % (name, str(detail)[:170]))


print("\nFree checks")
r = client.get("/health")
check("health is 200", r.status_code == 200, r.text)
check("health reports the model", r.json().get("model") == "gpt-6-luna", r.text)

r = client.get("/")
check("page renders", r.status_code == 200 and "AI Career Coach" in r.text)

r = client.post("/start", json={"role": "", "job_description": "", "focus": ""})
check("empty role -> 422", r.status_code == 422, r.text)

r = client.post("/start", json={"role": "x" * 200})
check("over-long role -> 422", r.status_code == 422, r.text)

r = client.post("/start", json={"role": "AI Engineer", "job_description": "x" * 9000})
check("over-long job description -> 422", r.status_code == 422, r.text)

r = client.post("/answer", json={"role": "AI Engineer", "turns": [], "answer": "hello"})
check("answer without the question -> 422", r.status_code == 422, r.text)

r = client.post("/answer", json={"role": "AI Engineer", "turns": [],
                                 "question": "q", "answer": ""})
check("empty answer -> 422", r.status_code == 422, r.text)

r = client.post("/finish", json={"role": "AI Engineer", "turns": []})
check("finish with no turns -> 422", r.status_code == 422, r.text)

print("\nCV review guards")
r = client.post("/cv", json={"role": "AI Engineer", "cv_text": "too short"})
check("short CV -> 422", r.status_code == 422, r.text)

r = client.post("/cv", json={"role": "", "cv_text": "x" * 500})
check("CV with no role -> 422", r.status_code == 422, r.text)

r = client.post("/cv", json={"role": "AI Engineer", "cv_text": "x" * 25000})
check("over-long CV -> 422", r.status_code == 422, r.text)

r = client.post("/cv/upload",
                files={"file": ("cv.exe", b"MZ binary", "application/octet-stream")},
                data={"role": "AI Engineer"})
check("unsupported file type -> 422", r.status_code == 422, r.text)

r = client.post("/cv/upload", files={"file": ("cv.pdf", b"", "application/pdf")},
                data={"role": "AI Engineer"})
check("empty file -> 422", r.status_code == 422, r.text)

r = client.post("/cv/upload", files={"file": ("cv.pdf", b"not a real pdf", "application/pdf")},
                data={"role": "AI Engineer"})
check("corrupt PDF -> 422 not 502", r.status_code == 422, r.text)

from api.coach import extract_cv_text                      # noqa: E402

check("txt extraction works",
      "Subha" in extract_cv_text(b"Name: Subha\nRole: Engineer", "cv.txt"))
try:
    extract_cv_text(b"x", "cv.png")
    check("png is refused", False, "no error raised")
except ValueError as error:
    check("png is refused", "Unsupported" in str(error), str(error))

# ----------------------------------------------------------- job fit, free
r = client.post("/jobfit", json={"role": "AI Engineer", "job_description": "",
                                 "cv_text": "x" * 500})
check("job fit without an advert -> 422", r.status_code == 422, r.text)
check("and says why", "advert" in r.text.lower(), r.text)

r = client.post("/jobfit", json={"role": "AI Engineer", "job_description": "Python.",
                                 "cv_text": "x" * 500})
check("job fit with a one-line advert -> 422", r.status_code == 422, r.text)

r = client.post("/jobfit", json={"role": "AI Engineer", "job_description": "y" * 300,
                                 "cv_text": "too short"})
check("job fit with a stub CV -> 422", r.status_code == 422, r.text)

r = client.post("/jobfit", json={"role": "AI Engineer", "job_description": "y" * 9000,
                                 "cv_text": "x" * 500})
check("job fit with an over-long advert -> 422", r.status_code == 422, r.text)

r = client.post("/jobfit/upload", files={"file": ("cv.txt", b"x" * 500, "text/plain")},
                data={"role": "AI Engineer"})
check("job fit upload without an advert -> 422", r.status_code == 422, r.text)

check("health advertises the job fit mode",
      "job fit" in client.get("/health").json().get("modes", []), client.get("/health").text)
check("the page offers the job fit button", "Job-fit analysis" in client.get("/").text)

# The score is arithmetic now, so it can be checked for free.
from api.coach import Requirement, score_fit                # noqa: E402


def req(must, status):
    return Requirement(requirement="r", must_have=must, status=status)


check("all requirements evidenced -> 10",
      score_fit([req(True, "evidenced"), req(False, "evidenced")])["score"] == 10)
check("nothing evidenced -> 1",
      score_fit([req(True, "missing"), req(False, "missing")])["score"] == 1)
check("no requirements -> no score", score_fit([])["score"] is None)

capped = score_fit([req(True, "evidenced")] * 4 + [req(True, "missing")])
check("a missing must-have caps the score at 7", capped["score"] == 7, capped["basis"])
check("and the cap names what blocks it", capped["basis"]["blocking"] == ["r"], capped["basis"])
check("the raw score before the cap is kept", capped["basis"]["raw_score"] > 7, capped["basis"])

check("a partial match earns half of an evidenced one",
      score_fit([req(True, "partial")])["basis"]["earned"] * 2
      == score_fit([req(True, "evidenced")])["basis"]["earned"])
check("a must-have outweighs a nice-to-have",
      score_fit([req(True, "evidenced")])["basis"]["possible"]
      > score_fit([req(False, "evidenced")])["basis"]["possible"])

same = [req(True, "evidenced"), req(True, "partial"), req(False, "missing")]
check("the same classification always scores the same",
      score_fit(same) == score_fit(same))

# -------------------------------------------------------- llm as judge, free
from api.coach import JUDGE_MODEL, MODEL                   # noqa: E402

check("the judge runs on a different model than the work",
      JUDGE_MODEL != MODEL, "%s vs %s" % (JUDGE_MODEL, MODEL))

h = client.get("/health").json()
check("health names the judge model", h.get("judge_model") == JUDGE_MODEL, h)
check("health lists the four criteria", len(h.get("judge_criteria", [])) == 4, h)
check("health lists the judge mode", "llm as a judge" in h.get("modes", []), h)

r = client.post("/judge", json={"cv_text": "x" * 500, "review": "not json"})
check("a review that is not JSON -> 422", r.status_code == 422, r.text)

r = client.post("/judge", json={"cv_text": "x" * 500, "review": "[1,2,3]"})
check("a review that is not an object -> 422", r.status_code == 422, r.text)

r = client.post("/judge", json={"cv_text": "x" * 500, "review": "{}"})
check("an empty review -> 422", r.status_code == 422, r.text)

r = client.post("/judge", json={"cv_text": "short", "review": '{"headline":"x"}'})
check("judging without the CV -> 422", r.status_code == 422, r.text)

r = client.post("/judge", json={"cv_text": "x" * 500})
check("no review at all -> 422", r.status_code == 422, r.text)

from api.coach import review_as_text                        # noqa: E402

flat = review_as_text("cv review", {"match_score": 7, "headline": "h",
                                    "green": [{"dimension": "content", "point": "p"}],
                                    "red": [], "evidenced_requirements": ["Python"],
                                    "missing_requirements": [], "quick_wins": []})
check("a review flattens to something readable",
      "7/10" in flat and "GREEN [content] p" in flat, flat)
check("job fit flattens too",
      "[evidenced]" in review_as_text("job fit", {
          "match_score": 6, "headline": "h",
          "requirements": [{"requirement": "Python", "must_have": True,
                            "status": "evidenced", "evidence": "five years",
                            "note": None}]}))
check("the page offers the judge",
      '"Have " + JUDGE + " judge this"' in client.get("/").text)
check("the page names the judge model", "gpt-6-astra" in client.get("/").text)

if "--live" not in sys.argv:
    print("\n%d passed, %d failed   (run with --live for a real interview)" % (passed, failed))
    sys.exit(1 if failed else 0)

print("\nLive CV review (this costs money)")
CV = """Subharanjan Chakraborty - Data and AI Consultant, Dusseldorf.
Twenty years of data work: Oracle and SQL Server warehouses, Informatica ETL and
dimensional modelling. Five years analysing telecom data in Python with Pandas and
NumPy at Telefonica O2, building KPI dashboards and Crystal Reports. Since 2021:
scikit-learn and TensorFlow at SPICED Academy; currently NLP and transformers at
Ironhack, including a DistilBERT sentiment classifier on 34,600 reviews.
Languages: German B2, English fluent. Education: B.Tech Electronics."""

r = client.post("/cv", json={
    "role": "Data & AI Consultant",
    "job_description": "Python, scikit-learn, SQL, Azure. Required: Master's degree, German C1.",
    "cv_text": CV})
check("cv review -> 200", r.status_code == 200, r.text)
if r.status_code == 200:
    d = r.json()
    check("match score within 1-10", 1 <= d["match_score"] <= 10, d)
    check("green and red both populated", bool(d["green"]) and bool(d["red"]), d)
    check("every point carries a dimension",
          all(p.get("dimension") for p in d["green"] + d["red"]), d)
    missing = " ".join(d["missing_requirements"]).lower()
    check("spots the German C1 gap", "c1" in missing or "german" in missing,
          d["missing_requirements"])
    print("       %d/10 | %d green, %d red | %s"
          % (d["match_score"], len(d["green"]), len(d["red"]), d["headline"][:90]))

print("\nLive job fit (this costs money)")
ADVERT = """AI Engineer (m/w/d), Dusseldorf.
Dein Profil:
- Mindestens 3 Jahre Berufserfahrung in Python (zwingend erforderlich)
- Erfahrung mit LLM-APIs (zwingend erforderlich)
- Sehr gute Deutschkenntnisse, mindestens C1 (zwingend erforderlich)
- Erfahrung mit Docker von Vorteil
"""

r = client.post("/jobfit", json={"role": "AI Engineer", "job_description": ADVERT,
                                 "cv_text": CV})
check("job fit -> 200", r.status_code == 200, r.text)
if r.status_code == 200:
    d = r.json()
    b = d["score_basis"]
    check("score within 1-10", 1 <= d["match_score"] <= 10, d["match_score"])
    check("requirements were extracted", len(d["requirements"]) >= 3, d["requirements"])
    check("every requirement carries a status",
          all(q["status"] in ("evidenced", "partial", "missing") for q in d["requirements"]))
    check("an evidenced requirement quotes the CV",
          all(q["evidence"] for q in d["requirements"] if q["status"] == "evidenced"),
          [q for q in d["requirements"] if q["status"] == "evidenced" and not q["evidence"]])
    check("a missing requirement quotes nothing",
          all(not q["evidence"] for q in d["requirements"] if q["status"] == "missing"),
          [q for q in d["requirements"] if q["status"] == "missing" and q["evidence"]])

    # The number on screen must be the number the arithmetic produced.
    raw = 1 + 9 * (b["earned"] / b["possible"])
    expected = min(round(raw), 7) if b["blocking"] else round(raw)
    check("the score recomputes from its own basis", d["match_score"] == expected,
          "%s vs %s (earned %s of %s)" % (d["match_score"], expected, b["earned"], b["possible"]))
    check("the basis counts every requirement",
          b["must_have"]["total"] + b["nice_to_have"]["total"] == len(d["requirements"]), b)

    # German B2 against a C1 requirement must not be waved through.
    german = [q for q in d["requirements"] if "c1" in q["requirement"].lower()
              or "deutsch" in q["requirement"].lower()]
    check("the German C1 bar is not marked evidenced",
          bool(german) and all(q["status"] != "evidenced" for q in german), german)
    check("gaps are listed worst first",
          [g["status"] for g in d["gaps"]] == sorted(
              [g["status"] for g in d["gaps"]],
              key=lambda st: st == "partial") if d["gaps"] else True, d["gaps"])
    print("       %d/10 | %d requirements, %d gaps | earned %s of %s | %s"
          % (d["match_score"], len(d["requirements"]), len(d["gaps"]),
             b["earned"], b["possible"], d["headline"][:70]))

print("\nLive judge (this costs money)")
# Every claim in this review is absent from CV above. A judge worth having
# says so; one that hands out sevens does not.
INVENTED = {
    "match_score": 9,
    "headline": "An outstanding fit with everything the advert asks for.",
    "green": [
        {"dimension": "relevant skills",
         "point": "Holds a Master's degree in Computer Science, exactly as required."},
        {"dimension": "relevant skills", "point": "German at C1 level."},
        {"dimension": "content", "point": "Six years of production Kubernetes at Telefonica."},
    ],
    "red": [{"dimension": "formatting", "point": "Could use more white space."}],
    "evidenced_requirements": ["German C1", "Master's degree", "Kubernetes"],
    "missing_requirements": [],
    "quick_wins": ["Lead with the Master's degree."],
}
JD = "AI Engineer. Required: Python, LLM APIs, German C1, a Master's degree."

r = client.post("/judge", json={"role": "AI Engineer", "job_description": JD,
                                "cv_text": CV, "review": json.dumps(INVENTED),
                                "kind": "cv review"})
check("judging a fabricated review -> 200", r.status_code == 200, r.text)
if r.status_code == 200:
    v = r.json()
    check("the verdict comes from the judge model", v["model"] == JUDGE_MODEL, v["model"])
    check("and says whose work it judged", v["judged_model"] == MODEL, v["judged_model"])
    check("all four criteria scored", len(v["criteria"]) == 4, v["criteria"])
    check("every criterion is justified",
          all(c["justification"].strip() for c in v["criteria"]))
    check("overall is the mean of the four, not a separate opinion",
          abs(v["overall"] - sum(c["score"] for c in v["criteria"]) / 4) < 0.06, v["overall"])
    check("the invented qualifications are caught",
          len(v["unsupported_claims"]) >= 2, v["unsupported_claims"])
    check("the German C1 fabrication specifically is caught",
          any("c1" in (u["claim"] + u["problem"]).lower() for u in v["unsupported_claims"]),
          v["unsupported_claims"])
    acc = [c["score"] for c in v["criteria"] if c["criterion"] == "factual accuracy"][0]
    check("factual accuracy is scored low for a review that invents things",
          acc <= 4, acc)
    print("       %s/10 overall | %d unsupported claims | factual accuracy %d/10 | %.0fs"
          % (v["overall"], len(v["unsupported_claims"]), acc, v["seconds"]))

print("\nLive interview (this costs money)")
JD = ("Agentic AI Engineer. You will build LLM agents that call tools, own small services "
      "end to end, and reason about latency and cost per request. Python, async, typing. "
      "Experience with an LLM API. Embeddings, vector search and RAG.")

r = client.post("/start", json={"role": "Agentic AI Engineer", "job_description": JD,
                                "focus": "tool calling"})
check("start -> 200", r.status_code == 200, r.text)
d = r.json()
print("       Q1:", d["question"][:150].replace("\n", " "))
turns = []
q = d["question"]

for answer in ["I've built agents that use tools and they work pretty well.",
               "In my travel advisor I defined a get_weather tool with a strict JSON schema, "
               "validated the model's arguments in Python before running it, and capped it at "
               "3 tool rounds so a loop couldn't run up the bill."]:
    r = client.post("/answer", json={"role": "Agentic AI Engineer", "job_description": JD,
                                     "focus": "tool calling", "turns": turns,
                                     "question": q, "answer": answer})
    check("answer -> 200", r.status_code == 200, r.text)
    d = r.json()
    turns.append({"question": q, "answer": answer})
    a = d.get("answer_score")
    check("answer was marked", a is not None, d)
    if a:
        print("       scored %d/10  good=%d bad=%d" % (a["score"], len(a["good"]), len(a["bad"])))
    q = d.get("question") or q
    if d.get("question"):
        print("       next:", d["question"][:130].replace("\n", " "))

r = client.post("/finish", json={"role": "Agentic AI Engineer", "turns": turns})
check("finish -> 200", r.status_code == 200, r.text)
f = r.json()["feedback"]
check("score is 1-10", 1 <= f["overall_score"] <= 10, f)
print("       overall %d/10 | %s" % (f["overall_score"], f["headline"][:110]))

print("\n%d passed, %d failed" % (passed, failed))
sys.exit(1 if failed else 0)
