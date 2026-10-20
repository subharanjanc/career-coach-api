"""Local checks. `python test_api.py` is free; `--live` runs a real interview."""

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
