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

if "--live" not in sys.argv:
    print("\n%d passed, %d failed   (run with --live for a real interview)" % (passed, failed))
    sys.exit(1 if failed else 0)

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
