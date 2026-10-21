"""FastAPI backend for the AI Career Coach - mock interview mode.

    GET  /          the interview page
    POST /start     role + job description -> first question
    POST /answer    an answer -> that answer marked, plus the next question
    POST /finish    the transcript -> overall score and feedback
    POST /cv        a CV -> a review against the role, and the advert if given
    POST /jobfit    a CV + an advert -> requirement-by-requirement fit, scored
    GET  /health    liveness

Stateless: the browser returns the transcript on every call.
"""

from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.responses import HTMLResponse
from openai import APITimeoutError
from pydantic import BaseModel, Field

try:
    from .coach import (MAX_CV_CHARS, MAX_JD_CHARS, MAX_QUESTIONS, MIN_JD_FOR_FIT,
                        MODEL, Turn, analyse_fit, extract_cv_text, final_feedback,
                        first_question, next_question, review_cv)
except ImportError:                                  # pragma: no cover
    from coach import (MAX_CV_CHARS, MAX_JD_CHARS, MAX_QUESTIONS, MIN_JD_FOR_FIT,
                       MODEL, Turn, analyse_fit, extract_cv_text, final_feedback,
                       first_question, next_question, review_cv)

app = FastAPI(title="AI Career Coach", version="1.0.0",
              description="Mock interview practice. Mini-project, Ironhack AI FT Sept 26.")


class StartRequest(BaseModel):
    role: str = Field(..., description="Target job title, e.g. Agentic AI Engineer")
    job_description: str = Field(default="", description="Optional job offer, pasted")
    focus: str = Field(default="", description="Optional: what to be pressed on")


class AnswerRequest(StartRequest):
    turns: list[Turn] = Field(
        default_factory=list,
        description="Completed question/answer pairs; empty on the first answer")
    question: str = Field(..., description="The question currently being answered")
    answer: str = Field(..., description="The candidate's answer to it")


class CVRequest(BaseModel):
    role: str = Field(..., description="Target job title")
    job_description: str = Field(default="", description="Optional advert to match against")
    cv_text: str = Field(..., description="The CV as plain text")


class FitRequest(BaseModel):
    role: str = Field(..., description="Target job title")
    job_description: str = Field(..., description="The advert - required for a fit analysis")
    cv_text: str = Field(..., description="The CV as plain text")


class FinishRequest(BaseModel):
    role: str = ""
    turns: list[Turn]


def _guard(fn, *args):
    try:
        return fn(*args)
    except ValueError as error:
        raise HTTPException(status_code=422, detail=str(error))
    except RuntimeError as error:
        raise HTTPException(status_code=503, detail=str(error))
    except APITimeoutError:
        raise HTTPException(status_code=504, detail="That took too long. Try a shorter answer.")
    except Exception as error:
        raise HTTPException(status_code=502, detail=f"Upstream failure ({type(error).__name__}).")


@app.post("/start")
def start(r: StartRequest):
    return _guard(first_question, r.role, r.job_description, r.focus)


@app.post("/answer")
def answer(r: AnswerRequest):
    return _guard(next_question, r.role, r.job_description, r.focus,
                  r.turns, r.question, r.answer)


@app.post("/finish")
def finish(r: FinishRequest):
    return _guard(final_feedback, r.role, r.turns)


@app.post("/cv")
def cv(r: CVRequest):
    """Review a pasted CV against the role and, if given, the advert."""
    return _guard(review_cv, r.role, r.job_description, r.cv_text)


@app.post("/cv/upload")
async def cv_upload(file: UploadFile = File(...),
                    role: str = Form(...),
                    job_description: str = Form("")):
    """Same review, but from an uploaded PDF, DOCX or text file."""
    data = await file.read()
    if len(data) > 4_000_000:
        raise HTTPException(status_code=422, detail="That file is larger than 4 MB.")
    try:
        text = extract_cv_text(data, file.filename or "")
    except ValueError as error:
        raise HTTPException(status_code=422, detail=str(error))
    return _guard(review_cv, role, job_description, text)


@app.post("/jobfit")
def jobfit(r: FitRequest):
    """Score a CV against one advert, requirement by requirement."""
    return _guard(analyse_fit, r.role, r.job_description, r.cv_text)


@app.post("/jobfit/upload")
async def jobfit_upload(file: UploadFile = File(...),
                        role: str = Form(...),
                        job_description: str = Form(...)):
    """Same analysis, but from an uploaded PDF, DOCX or text file."""
    data = await file.read()
    if len(data) > 4_000_000:
        raise HTTPException(status_code=422, detail="That file is larger than 4 MB.")
    try:
        text = extract_cv_text(data, file.filename or "")
    except ValueError as error:
        raise HTTPException(status_code=422, detail=str(error))
    return _guard(analyse_fit, role, job_description, text)


@app.get("/health")
def health():
    return {"status": "ok", "model": MODEL, "max_questions": MAX_QUESTIONS,
            "max_job_description_chars": MAX_JD_CHARS,
            "max_cv_chars": MAX_CV_CHARS,
            "min_job_description_chars_for_fit": MIN_JD_FOR_FIT,
            "modes": ["mock interview", "cv review", "job fit"]}


PAGE = """<!doctype html>
<html lang="en"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>AI Career Coach</title>
<style>
  :root { --bg:#0B1524; --card:#13223A; --card2:#1A2D4A; --ink:#E8F1FB;
          --muted:#8FA9C7; --line:#27405F; --accent:#4A8FD6; --accent2:#7FBCF5;
          --good:#3FBF72; --bad:#E2615A; --warn:#E0B341; }
  * { box-sizing:border-box }
  body { margin:0; background:var(--bg); color:var(--ink);
         font:16px/1.6 system-ui,sans-serif }
  .wrap { max-width:780px; margin:0 auto; padding:26px 16px 140px }
  h1 { font-size:22px; font-weight:600; margin:0 0 4px }
  h1 span { color:var(--accent2) }
  .sub { color:var(--muted); font-size:13.5px; margin-bottom:20px }
  label { display:block; font-size:12px; color:var(--muted); text-transform:uppercase;
          letter-spacing:.06em; margin:14px 0 5px }
  input, textarea { width:100%; padding:11px 13px; border:1px solid var(--line);
         border-radius:9px; background:var(--card); color:var(--ink);
         font:15px/1.5 system-ui,sans-serif; resize:vertical }
  input:focus, textarea:focus { outline:0; border-color:var(--accent);
         box-shadow:0 0 0 3px rgba(74,143,214,.2) }
  textarea { min-height:120px }
  button { padding:11px 22px; border:0; border-radius:9px; background:var(--accent);
           color:#fff; font-size:15px; font-weight:600; cursor:pointer }
  button:hover { background:var(--accent2); color:#0B1524 }
  button:disabled { background:var(--line); color:var(--muted); cursor:default }
  button.ghost { background:transparent; border:1px solid var(--line); color:var(--muted) }
  .msg { background:var(--card); border:1px solid var(--line); border-radius:12px;
         padding:13px 16px; margin:8px 0; white-space:pre-wrap; overflow-wrap:anywhere }
  .me { background:transparent; border:1px dashed var(--accent) }
  .who { font-size:11px; color:var(--muted); text-transform:uppercase;
         letter-spacing:.06em; margin-bottom:5px }
  .card { border:1px solid var(--line); border-left:4px solid var(--warn);
          border-radius:10px; padding:11px 14px; margin:8px 0; background:var(--card2);
          font-size:14px }
  .card .hd { font-size:11px; color:var(--muted); text-transform:uppercase;
              letter-spacing:.06em; margin-bottom:6px }
  .g { color:var(--good) } .b { color:var(--bad) }
  .fix { margin-top:8px; padding-top:7px; border-top:1px solid var(--line);
         color:var(--accent2) }
  .meta { font-size:12px; color:var(--muted); margin:2px 0 12px 4px }
  .bar { position:fixed; left:0; right:0; bottom:0; background:var(--bg);
         border-top:1px solid var(--line); padding:12px 16px }
  .bar div { max-width:780px; margin:0 auto; display:flex; gap:8px }
  .bar textarea { min-height:46px; max-height:160px }
  .hidden { display:none }
  .big { font-size:30px; font-weight:600 }
  .rev { border:1px solid var(--line); border-radius:12px; padding:14px 16px; margin:10px 0;
         background:var(--card) }
  .rev h3 { font-size:14px; margin:12px 0 6px; font-weight:600 }
  .pt { margin:4px 0; font-size:14px; line-height:1.5 }
  .pt .dim { font-size:11px; text-transform:uppercase; letter-spacing:.05em;
             color:var(--muted); margin-right:6px }
  .pill { display:inline-block; font-size:12px; padding:3px 9px; border-radius:999px;
          margin:3px 4px 0 0; background:var(--card2); color:var(--muted) }
  .pill.ok { color:var(--good) } .pill.no { color:var(--bad) }
  .row { display:flex; gap:8px; flex-wrap:wrap; margin-top:16px }
  input[type=file] { padding:9px; background:var(--card) }
  .req { display:flex; gap:9px; align-items:flex-start; padding:7px 0;
         border-bottom:1px solid var(--line); font-size:14px }
  .req:last-child { border-bottom:0 }
  .req .mk { flex:0 0 18px; line-height:1.5; text-align:center }
  .req .rq { font-weight:600 }
  .req .ev { color:var(--muted); font-size:12.5px; display:block; margin-top:2px }
  .req .nt { color:var(--accent2); font-size:12.5px; display:block; margin-top:2px }
  .tag { font-size:10px; text-transform:uppercase; letter-spacing:.05em;
         padding:1px 6px; border-radius:4px; background:var(--card2);
         color:var(--muted); margin-left:7px; vertical-align:1px }
  .sum { font-size:12.5px; color:var(--muted); background:var(--card2);
         border-radius:8px; padding:9px 12px; margin:10px 0; line-height:1.65 }
  .sum b { color:var(--ink) }
</style></head><body>
<div class="wrap">
  <h1>AI Career Coach<span>.</span></h1>
  <div class="sub">Paste the job you are preparing for. It interviews you one question at a
    time, marks every answer, and scores the whole thing at the end. Or have your CV
    reviewed, or matched against the advert requirement by requirement.</div>

  <div id="setup">
    <label for="role">Role you are preparing for</label>
    <input id="role" placeholder="Agentic AI Engineer">
    <label for="focus">What should it press you on? (optional)</label>
    <input id="focus" placeholder="tool calling, system design">
    <label for="jd">Job description (optional, sharper questions)</label>
    <textarea id="jd" placeholder="Paste the advert here..."></textarea>
    <label for="cvfile">Your CV &mdash; upload (PDF, DOCX, TXT)</label>
    <input id="cvfile" type="file" accept=".pdf,.docx,.txt,.md">
    <label for="cvtext">&hellip;or paste it instead</label>
    <textarea id="cvtext" placeholder="Paste your CV text here..."></textarea>

    <div class="row">
      <button id="review">Review my CV</button>
      <button id="fit" class="ghost">Job-fit analysis</button>
      <button id="go" class="ghost">Skip and start interview</button>
    </div>
  </div>

  <div id="cvout"></div>

  <div id="log"></div>
</div>

<div class="bar hidden" id="bar"><div>
  <textarea id="ans" placeholder="Your answer..."></textarea>
  <button id="send">Send</button>
  <button id="fin" class="ghost">Finish</button>
</div></div>

<script>
const S = { role:"", jd:"", focus:"", turns:[], asked:0, lastQ:"", spent:0, done:false };
const $ = id => document.getElementById(id);
const log = $("log");

function esc(t){ return (t||"").replace(/&/g,"&amp;").replace(/</g,"&lt;").replace(/\\*\\*/g,""); }
function bubble(who, text, mine){
  const d = document.createElement("div");
  d.className = "msg" + (mine ? " me" : "");
  d.innerHTML = "<div class='who'>" + who + "</div>" + esc(text);
  log.appendChild(d); scroll(); return d;
}
function meta(t){
  const d = document.createElement("div"); d.className="meta"; d.textContent=t;
  log.appendChild(d); scroll();
}
function scroll(){ window.scrollTo(0, document.body.scrollHeight); }

function scoreCard(a){
  if(!a) return;
  const hue = a.score>=7 ? "var(--good)" : (a.score>=5 ? "var(--warn)" : "var(--bad)");
  let rows = "";
  (a.good||[]).forEach(x => rows += "<div class='g'>+ " + esc(x) + "</div>");
  (a.bad||[]).forEach(x => rows += "<div class='b'>&minus; " + esc(x) + "</div>");
  const fix = a.one_fix ? "<div class='fix'><b>Fix next time:</b> " + esc(a.one_fix) + "</div>" : "";
  const d = document.createElement("div");
  d.className = "card"; d.style.borderLeftColor = hue;
  d.innerHTML = "<div class='hd'>your answer &middot; <b style='color:"+hue+";font-size:13px'>"
              + a.score + "/10</b></div>" + rows + fix;
  log.appendChild(d); scroll();
}

async function post(path, body){
  const r = await fetch(path, { method:"POST", headers:{"Content-Type":"application/json"},
                                body: JSON.stringify(body) });
  const data = await r.json();
  if(!r.ok) throw new Error(data.detail || ("HTTP " + r.status));
  return data;
}

function waiting(text){
  const d = document.createElement("div"); d.className="meta"; d.textContent = text + " 0s";
  log.appendChild(d); scroll();
  const t0 = Date.now();
  const id = setInterval(()=>{ d.textContent = text + " " + Math.round((Date.now()-t0)/1000) + "s"; }, 1000);
  return () => { clearInterval(id); d.remove(); };
}

function reviewCard(d){
  const hue = d.match_score >= 7 ? "var(--good)" : (d.match_score >= 5 ? "var(--warn)" : "var(--bad)");
  let h = "<div class='who'>cv review</div>";
  h += "<div class='big' style='color:" + hue + "'>" + d.match_score + "/10</div>";
  h += "<div style='margin-bottom:8px'>" + esc(d.headline) + "</div>";

  if(d.green.length){
    h += "<h3 class='g'>What works</h3>";
    d.green.forEach(p => h += "<div class='pt g'><span class='dim'>" + esc(p.dimension)
                           + "</span>" + esc(p.point) + "</div>");
  }
  if(d.red.length){
    h += "<h3 class='b'>What needs improving</h3>";
    d.red.forEach(p => h += "<div class='pt b'><span class='dim'>" + esc(p.dimension)
                          + "</span>" + esc(p.point) + "</div>");
  }
  if(d.evidenced_requirements.length || d.missing_requirements.length){
    h += "<h3>Against the advert</h3><div>";
    d.evidenced_requirements.forEach(x => h += "<span class='pill ok'>&#10003; " + esc(x) + "</span>");
    d.missing_requirements.forEach(x => h += "<span class='pill no'>&#10007; " + esc(x) + "</span>");
    h += "</div>";
  }
  if(d.quick_wins.length){
    h += "<h3>Quick wins</h3>";
    d.quick_wins.forEach(x => h += "<div class='pt'>&bull; " + esc(x) + "</div>");
  }
  const el = document.createElement("div");
  el.className = "rev"; el.style.borderLeft = "4px solid " + hue; el.innerHTML = h;
  $("cvout").appendChild(el);
  const b = document.createElement("button");
  b.textContent = "Start the interview"; b.style.marginTop = "12px";
  b.onclick = () => { $("cvout").innerHTML = ""; $("go").click(); };
  $("cvout").appendChild(b);
  scroll();
}

// /cv and /jobfit take the same CV in the same two ways, so the
// upload-or-paste dance lives here once instead of once per button.
async function postCV(jsonPath, uploadPath){
  const role = $("role").value.trim();
  if(!role) throw new Error("Tell it which role first.");
  const f = $("cvfile").files[0];
  const pasted = $("cvtext").value.trim();
  if(!f && !pasted) throw new Error("Upload a CV or paste the text.");
  const jd = $("jd").value.trim();
  if(f){
    const fd = new FormData();
    fd.append("file", f); fd.append("role", role); fd.append("job_description", jd);
    const r = await fetch(uploadPath, { method:"POST", body: fd });
    const d = await r.json();
    if(!r.ok) throw new Error(d.detail || ("HTTP " + r.status));
    return d;
  }
  return await post(jsonPath, { role: role, job_description: jd, cv_text: pasted });
}

function busy(on){ ["review","fit","go"].forEach(id => { $(id).disabled = on; }); }

async function runCV(label, jsonPath, uploadPath, render){
  busy(true); $("cvout").innerHTML = "";
  const stop = waiting(label);
  try{
    const d = await postCV(jsonPath, uploadPath);
    stop(); S.spent += d.estimated_cost_usd; render(d);
  }catch(e){ stop(); bubble("error", e.message); }
  busy(false);
}

$("review").onclick = () =>
  runCV("reading your CV against the role...", "/cv", "/cv/upload", reviewCard);

$("fit").onclick = () => {
  // The server refuses a fit analysis without an advert; saying so here saves
  // the round trip. The threshold is the server's own, not a second opinion.
  if($("jd").value.trim().length < """ + str(MIN_JD_FOR_FIT) + """){
    alert("A job-fit analysis needs the advert - that is where the requirements are "
        + "read from. Paste the job description above, then try again.");
    return;
  }
  runCV("matching your CV against the advert...", "/jobfit", "/jobfit/upload", fitCard);
};

const MARK = { evidenced:["&#10003;","var(--good)"], partial:["&#126;","var(--warn)"],
               missing:["&#10007;","var(--bad)"] };

function reqRow(q){
  const m = MARK[q.status] || MARK.missing;
  let h = "<div class='req'><div class='mk' style='color:" + m[1] + "'>" + m[0] + "</div><div>";
  h += "<span class='rq'>" + esc(q.requirement) + "</span>";
  h += "<span class='tag'>" + (q.must_have ? "must have" : "nice to have") + "</span>";
  if(q.evidence) h += "<span class='ev'>CV: " + esc(q.evidence) + "</span>";
  if(q.note)     h += "<span class='nt'>" + esc(q.note) + "</span>";
  return h + "</div></div>";
}

function fitCard(d){
  const sc = d.match_score;
  const hue = sc >= 7 ? "var(--good)" : (sc >= 5 ? "var(--warn)" : "var(--bad)");
  let h = "<div class='who'>job fit against the advert</div>";
  h += "<div class='big' style='color:" + hue + "'>"
     + (sc === null ? "&mdash;" : sc + "/10") + "</div>";
  h += "<div style='margin-bottom:4px'>" + esc(d.headline) + "</div>";

  // The score is arithmetic, so show the arithmetic. A number you cannot
  // recompute is a number you cannot defend in a real interview.
  const b = d.score_basis;
  if(b){
    h += "<div class='sum'>Counted, not guessed: <b>" + b.earned + " of " + b.possible
       + "</b> weighted points (a must-have counts " + b.weights.must_have
       + ", a nice-to-have " + b.weights.nice_to_have
       + "; a partial match earns half) &rarr; <b>" + b.raw_score + "</b>";
    if(b.capped_at !== null && b.capped_at !== undefined){
      h += ", held down to <b>" + b.capped_at + "</b> because a must-have is missing: "
         + esc(b.blocking.join(", "));
    }
    h += ".<br>Must-haves: <b>" + b.must_have.evidenced + "</b> evidenced, <b>"
       + b.must_have.partial + "</b> partial, <b>" + b.must_have.missing
       + "</b> missing, of " + b.must_have.total;
    if(b.nice_to_have.total){
      h += ". Nice-to-haves: <b>" + b.nice_to_have.evidenced + "</b> of "
         + b.nice_to_have.total + " evidenced";
    }
    h += ".</div>";
  }

  const musts = (d.requirements || []).filter(q => q.must_have);
  const nices = (d.requirements || []).filter(q => !q.must_have);
  if(musts.length){ h += "<h3>Required by the advert</h3>"; musts.forEach(q => h += reqRow(q)); }
  if(nices.length){ h += "<h3>Preferred</h3>"; nices.forEach(q => h += reqRow(q)); }

  if(d.gaps && d.gaps.length){
    h += "<h3 class='b'>Your skills gap, worst first</h3>";
    d.gaps.forEach(g => h += "<div class='pt'><span class='"
      + (g.status === "missing" ? "b" : "") + "'>"
      + (g.status === "missing" ? "&#10007; " : "&#126; ") + esc(g.requirement) + "</span>"
      + "<span class='tag'>" + (g.must_have ? "must have" : "nice to have") + "</span>"
      + (g.note ? "<span class='nt'>" + esc(g.note) + "</span>" : "") + "</div>");
  }
  if(d.transferable && d.transferable.length){
    h += "<h3 class='g'>Strengths the advert did not ask for</h3>";
    d.transferable.forEach(x => h += "<div class='pt g'>+ " + esc(x) + "</div>");
  }
  if(d.improvements && d.improvements.length){
    h += "<h3>What would raise the score</h3>";
    d.improvements.forEach(x => h += "<div class='pt'>&bull; " + esc(x) + "</div>");
  }

  const el = document.createElement("div");
  el.className = "rev"; el.style.borderLeft = "4px solid " + hue; el.innerHTML = h;
  $("cvout").appendChild(el);

  const next = document.createElement("button");
  next.textContent = "Interview me on the gaps"; next.style.marginTop = "12px";
  next.onclick = () => {
    // Walk the gap straight into practice: the must-haves it could not find are
    // exactly what a real interviewer would dig at.
    const gaps = (d.gaps || []).filter(g => g.must_have).map(g => g.requirement);
    if(gaps.length && !$("focus").value.trim()) $("focus").value = gaps.slice(0, 3).join(", ");
    $("cvout").innerHTML = ""; $("go").click();
  };
  $("cvout").appendChild(next);
  scroll();
}

$("go").onclick = async () => {
  S.role = $("role").value.trim(); S.focus = $("focus").value.trim(); S.jd = $("jd").value.trim();
  if(!S.role){ alert("Tell it which role first."); return; }
  $("go").disabled = true;
  const stop = waiting("preparing your interview...");
  try{
    const d = await post("/start", { role:S.role, job_description:S.jd, focus:S.focus });
    stop(); $("setup").classList.add("hidden"); $("cvout").innerHTML = "";
    $("bar").classList.remove("hidden");
    S.lastQ = d.question; S.asked = d.asked; S.spent += d.estimated_cost_usd;
    bubble("interviewer", d.question);
    meta("question " + d.asked + " of """ + str(MAX_QUESTIONS) + """ \\u00b7 $" + S.spent.toFixed(5));
    $("ans").focus();
  }catch(e){ stop(); $("go").disabled=false; bubble("error", e.message); }
};

$("send").onclick = async () => {
  const a = $("ans").value.trim();
  if(!a || S.done) return;
  $("ans").value = ""; $("send").disabled = true; $("fin").disabled = true;
  bubble("you", a, true);
  const stop = waiting("marking your answer...");
  try{
    const d = await post("/answer", { role:S.role, job_description:S.jd, focus:S.focus,
                                      turns:S.turns, question:S.lastQ, answer:a });
    stop();
    S.turns.push({ question:S.lastQ, answer:a });
    S.spent += d.estimated_cost_usd;
    scoreCard(d.answer_score);
    if(d.done || !d.question){ await finish(); }
    else {
      S.lastQ = d.question; S.asked = d.asked;
      bubble("interviewer", d.question);
      meta("question " + d.asked + " of """ + str(MAX_QUESTIONS) + """ \\u00b7 $" + S.spent.toFixed(5));
    }
  }catch(e){ stop(); bubble("error", e.message); }
  $("send").disabled = false; $("fin").disabled = false; $("ans").focus();
};

function readyForNextRun(){
  // Keep the transcript and the feedback on screen; bring the setup back under
  // it so another CV or another interview is one click, with the role, advert
  // and CV still filled in. A page reload would discard all of that.
  S.turns = []; S.asked = 0; S.lastQ = ""; S.done = false;
  $("bar").classList.add("hidden");
  $("setup").classList.remove("hidden");
  $("review").disabled = false;
  $("go").disabled = false;
  const note = document.createElement("div");
  note.className = "meta";
  note.textContent = "Review another CV, or start a fresh interview below. "
                   + "Your role, advert and CV are still filled in.";
  log.appendChild(note);
  $("setup").scrollIntoView({ behavior: "smooth", block: "start" });
}

async function finish(){
  if(S.done) return;
  if(!S.turns.length){ bubble("coach", "Answer at least one question first."); return; }
  S.done = true; $("send").disabled = true; $("fin").disabled = true;
  const stop = waiting("scoring the whole interview...");
  try{
    const d = await post("/finish", { role:S.role, turns:S.turns });
    stop(); S.spent += d.estimated_cost_usd;
    const f = d.feedback;
    let h = "<div class='who'>final feedback</div>";
    h += "<div class='big'>" + f.overall_score + "/10</div><div style='margin-bottom:10px'>"
       + esc(f.headline) + "</div>";
    if(f.strengths.length){ h += "<b>What went well</b>";
      f.strengths.forEach(x => h += "<div class='g'>+ " + esc(x) + "</div>"); }
    if(f.weaknesses.length){ h += "<b style='display:block;margin-top:10px'>What let you down</b>";
      f.weaknesses.forEach(x => h += "<div class='b'>&minus; " + esc(x) + "</div>"); }
    if(f.improvements.length){ h += "<b style='display:block;margin-top:10px'>Do this before the real one</b>";
      f.improvements.forEach(x => h += "<div>&bull; " + esc(x) + "</div>"); }
    if(f.would_pass_screen !== null && f.would_pass_screen !== undefined){
      h += "<div class='fix'>Would pass a first screen: " + (f.would_pass_screen ? "yes" : "not yet") + "</div>"; }
    const d2 = document.createElement("div"); d2.className = "msg"; d2.innerHTML = h;
    log.appendChild(d2);
    meta(d.questions_answered + " answers \\u00b7 $" + S.spent.toFixed(5) + " total");
    readyForNextRun();
  }catch(e){ stop(); S.done = false; $("send").disabled=false; $("fin").disabled=false;
             bubble("error", e.message); }
}
$("fin").onclick = finish;
$("ans").addEventListener("keydown", e => {
  if(e.key === "Enter" && !e.shiftKey){ e.preventDefault(); $("send").click(); }
});
</script></body></html>"""


@app.get("/", response_class=HTMLResponse)
def home():
    return PAGE
