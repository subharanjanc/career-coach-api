# AI Career Coach & CV Evaluation

Mini-project, Ironhack AI Engineering FT Sept 26.
Brief: `ironhack-rmt-AI-student-materials/project-brief-AI-powered-career-coach`

Paste the job you are preparing for. It interviews you **one question at a time**,
marks every answer out of 10, and scores the whole interview at the end. It also
reviews your CV, and matches it against the advert requirement by requirement.

## Endpoints

| Method | Path | What it does |
|---|---|---|
| `GET` | `/` | The interview page |
| `POST` | `/start` | Role + optional job description → the first question |
| `POST` | `/answer` | An answer → that answer marked, plus the next question |
| `POST` | `/finish` | The transcript → overall score, strengths, weaknesses, fixes |
| `POST` | `/cv` | A pasted CV → match score, green points, red points |
| `POST` | `/cv/upload` | A PDF, DOCX, TXT or MD file → the same review |
| `POST` | `/jobfit` | A CV **and an advert** → every requirement judged, then scored |
| `POST` | `/jobfit/upload` | A file + an advert → the same analysis |
| `POST` | `/judge` | A review → a second model's verdict on it |
| `POST` | `/judge/upload` | The same, when the CV was a file |
| `GET` | `/health` | Liveness, model, limits |

### `POST /answer`

```json
{
  "role": "Agentic AI Engineer",
  "job_description": "...optional...",
  "focus": "tool calling",
  "turns": [{ "question": "...", "answer": "..." }],
  "question": "the question being answered now",
  "answer": "the candidate's answer"
}
```

```json
{
  "question": "The round cap limits runaway calls, but doesn't explain failure handling...",
  "answer_score": {
    "score": 5,
    "good": ["Names a specific agent and tool"],
    "bad": ["Doesn't say what the validation actually checked"],
    "one_fix": "State one concrete unsafe argument the validation would reject."
  },
  "asked": 2,
  "done": false,
  "estimated_cost_usd": 0.00041
}
```

`turns` is the **completed** history and is empty on the first answer; `question`
is the one being answered now. Both are needed — see the design note below.

## Design notes

**Stateless.** The server keeps no session; the browser returns the transcript on
every call. A serverless function has nowhere to hold a conversation, and an
interview that dies because a container was recycled is worse than a slightly
larger request body.

**The question travels with the answer.** An earlier version derived it from
`turns[-1]`, which rejected the very first answer — at that point one question has
been asked and answered, and nothing is *complete* yet. The test suite caught it.

**One question at a time.** The prompt forbids asking two, and forbids the model
answering its own question however the candidate phrases the request.

**No scoring during the interview, by the interviewer.** Marking is a separate
call against a Pydantic model, so the interviewer never sees a grade and cannot
soften its next question because of one.

**Honesty is not punished.** The marking prompt says an honest "I don't know"
scores around 4 with the candour noted under `good`, rather than a 1.

**Marking never breaks the interview.** The scoring call is wrapped: if it fails
you lose the card, not the session.

**`would_pass_screen` is `bool | None`.** With too little signal it returns null
rather than guessing — a required `bool` would always pick one.

## Run locally

```bash
pip install -r requirements.txt
set OPENAI_API_KEY=sk-...
uvicorn api.index:app --reload
```

Open http://127.0.0.1:8000

## Tests

```bash
python test_api.py          # 9 checks, no API calls, free
python test_api.py --live   # adds a full interview: start, two answers, scoring
```

## Deploy

Vercel, from this repo. Set `OPENAI_API_KEY` in the project's environment
variables **before** the first deploy — without it the build still succeeds,
`/health` works, and every interview call returns `503`.

## CV review

Runs before the interview: role → job advert → CV → questions. It can be skipped.

`CVReview` is a Pydantic model, so the shape is guaranteed:

- `match_score` 1–10, and a one-sentence headline
- `green` and `red` — each point tagged with a dimension the brief names:
  content, clarity, formatting, errors, relevant skills
- `evidenced_requirements` / `missing_requirements` — split straight from the advert
- `quick_wins` — changes doable in under an hour

Extraction uses `pypdf` and `python-docx`, including DOCX tables, because CVs
often keep dates in them. A scanned PDF with no text layer is refused with an
explanation rather than silently reviewed as empty.

The prompt forbids assuming experience that is not written down: an unevidenced
requirement goes to `missing_requirements` rather than being charitably assumed.
A reviewer that invents qualifications is worse than no reviewer.

## Job fit and skills gap

Bonus challenge 2 of the brief: compare the CV against a specific offer, name
the relevant and missing skills, give a job-match score, suggest improvements.

The CV reviewer already produced all four, but its `match_score` was a holistic
judgement and drifted — 5, 5, 4, 5 across four identical requests. A score that
moves on its own is not a measurement.

So `/jobfit` splits the work. The model only **classifies**: it pulls each
requirement out of the advert, marks it `must_have` or not, and judges it
`evidenced` / `partial` / `missing` against the CV, quoting the CV as evidence.
The **score is arithmetic**, computed in `score_fit()`:

- a must-have is worth 3 points, a nice-to-have 1
- evidenced earns full credit, partial a half, missing nothing
- `score = 1 + 9 × earned / possible`, so the range is 1–10
- while any must-have is outright missing the score is **capped at 7** — a CV is
  not a strong fit for a job whose hard requirement it never mentions

Every number is returned in `score_basis` and shown on the page, so the score can
be recomputed by hand. A test asserts the returned score equals the one its own
basis implies.

### Does it actually hold still?

Three identical runs against a German advert, after tightening the prompt so the
advert is split the same way each time:

| | requirements | weighted total | score |
|---|---|---|---|
| run 1 | 13 | 29 | 5 |
| run 2 | 13 | 29 | 5 |
| run 3 | 13 | 29 | 6 |

The denominator and the blocking list are now stable; the score still moves by a
point when a borderline requirement flips between `partial` and `evidenced`.
That is honest: the spread is the same ±1 as before, but it is no longer
unexplained — `score_basis` says exactly which requirement moved and what it was
worth. Pinning it further would mean pretending a judgement call is not one.

A fit analysis without an advert is refused rather than scored, because there is
nothing to measure against. Use `/cv` for that.

## LLM as a judge

Bonus challenge 5: a model assesses the CV reviews against factual accuracy,
relevance, actionability and completeness, returning structured scores and
justifications.

It runs on **a different model on purpose**. `gpt-6-luna` does the work;
`gpt-6-astra` grades it. A model marking its own homework agrees with itself,
and the failure worth catching here — a review asserting something the CV never
said — is exactly the one its author is blind to.

The judge is handed the CV as ground truth and told it is grading the write-up,
not the candidate. `overall` is the mean of the four criteria rather than a
fifth number the judge was asked for: a model asked for an overall alongside its
parts tends to pick the overall first and bend the parts to fit.

### Does it discriminate, or just hand out sevens?

Three reviews of the same CV, judged by `gpt-6-astra`:

| Review | Overall | Factual accuracy | Unsupported claims caught |
|---|---|---|---|
| Genuine, written by `gpt-6-luna` | 9.2 | 10/10 | none |
| Fabricated — invents a Master's, German C1, six years of Kubernetes | 2.2 | 1/10 | **8** |
| True but useless — "improve the profile", "could be stronger" | 4.0 | 10/10 | none |

The third row is the one that shows the criteria are doing separate work. That
review is not wrong about anything, so factual accuracy is 10/10 — and
actionability is 1/10, which is why it still lands at 4.

On the fabricated review it quoted the claim and the contradiction: *"German at
C1 level, so the language bar is cleared"* against *"The CV explicitly states
German B2, not C1."* The genuine review drew no false accusations.

## Which model did what

Every response carries the model that produced it, and every card on the page
prints it. With two models in play, a verdict means very different things
depending on whether it came from the model that wrote the thing being judged —
so the page names both: *"gpt-6-astra judging what gpt-6-luna wrote"*.

## Brief status

| Requirement | State |
|---|---|
| 1. Mock interview mode | done, deployed |
| 2. CV reviewer mode | done, deployed |
| 3. One app containing both | done |
| Bonus 2 — job fit & skills gap | done, deployed |
| Bonus 1 — job search & discovery | not started |
| Bonus 3 — personalised learning roadmap | not started |
| Bonus 4 — company-specific interview prep | not started |
| Bonus 5 — LLM-as-a-judge evaluation | done, deployed |

## Limits

- A transcript is resent on every turn, so billed input grows through the
  interview. Each answer costs roughly $0.0004 (two calls: marking, then the
  next question).
- Eight questions maximum per interview.
- Job descriptions are capped at 6,000 characters and answers at 4,000.
