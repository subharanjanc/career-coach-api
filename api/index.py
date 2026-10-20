"""FastAPI backend for the AI Career Coach - mock interview mode.

    GET  /          the interview page
    POST /start     role + job description -> first question
    POST /answer    an answer -> that answer marked, plus the next question
    POST /finish    the transcript -> overall score and feedback
    GET  /health    liveness

Stateless: the browser returns the transcript on every call.
"""

from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.responses import HTMLResponse
from openai import APITimeoutError
from pydantic import BaseModel, Field

try:
    from .coach import (MAX_CV_CHARS, MAX_JD_CHARS, MAX_QUESTIONS, MODEL, Turn,
                        extract_cv_text, final_feedback, first_question,
                        next_question, review_cv)
except ImportError:                                  # pragma: no cover
    from coach import (MAX_CV_CHARS, MAX_JD_CHARS, MAX_QUESTIONS, MODEL, Turn,
                       extract_cv_text, final_feedback, first_question,
                       next_question, review_cv)

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


@app.get("/health")
def health():
    return {"status": "ok", "model": MODEL, "max_questions": MAX_QUESTIONS,
            "max_job_description_chars": MAX_JD_CHARS,
            "max_cv_chars": MAX_CV_CHARS,
            "modes": ["mock interview", "cv review"]}


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
</style></head><body>
<div class="wrap">
  <h1>AI Career Coach<span>.</span></h1>
  <div class="sub">Paste the job you are preparing for. It interviews you one question at a
    time, marks every answer, and scores the whole thing at the end.</div>

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

$("review").onclick = async () => {
  const role = $("role").value.trim();
  if(!role){ alert("Tell it which role first."); return; }
  const f = $("cvfile").files[0];
  const pasted = $("cvtext").value.trim();
  if(!f && !pasted){ alert("Upload a CV or paste the text."); return; }
  $("review").disabled = true; $("go").disabled = true;
  $("cvout").innerHTML = "";
  const stop = waiting("reading your CV against the role...");
  try{
    let d;
    if(f){
      const fd = new FormData();
      fd.append("file", f);
      fd.append("role", role);
      fd.append("job_description", $("jd").value.trim());
      const r = await fetch("/cv/upload", { method:"POST", body: fd });
      d = await r.json();
      if(!r.ok) throw new Error(d.detail || ("HTTP " + r.status));
    } else {
      d = await post("/cv", { role: role, job_description: $("jd").value.trim(),
                              cv_text: pasted });
    }
    stop(); S.spent += d.estimated_cost_usd; reviewCard(d);
  }catch(e){ stop(); bubble("error", e.message); }
  $("review").disabled = false; $("go").disabled = false;
};

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
