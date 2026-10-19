# AI Career Coach — mock interview mode

Mini-project, Ironhack AI Engineering FT Sept 26.
Brief: `ironhack-rmt-AI-student-materials/project-brief-AI-powered-career-coach`

Paste the job you are preparing for. It interviews you **one question at a time**,
marks every answer out of 10, and scores the whole interview at the end.

## Endpoints

| Method | Path | What it does |
|---|---|---|
| `GET` | `/` | The interview page |
| `POST` | `/start` | Role + optional job description → the first question |
| `POST` | `/answer` | An answer → that answer marked, plus the next question |
| `POST` | `/finish` | The transcript → overall score, strengths, weaknesses, fixes |
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

## Not built yet

- **CV reviewer mode** — upload or paste a CV, feedback on content, clarity,
  formatting, errors, relevant skills, improvements
- Both modes in one app, which is what the brief ultimately asks for

## Limits

- A transcript is resent on every turn, so billed input grows through the
  interview. Each answer costs roughly $0.0004 (two calls: marking, then the
  next question).
- Eight questions maximum per interview.
- Job descriptions are capped at 6,000 characters and answers at 4,000.
