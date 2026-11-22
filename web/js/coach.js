// Keys101: the page of scripts/keys_coach.py. It shows the coach's state and sends your choices back.
// Everything it shows comes from the server as one JSON state (GET /events). It sends: a lesson or a mode to
// start, stop, a key you clicked (POST /cmd). Two screens that slide past each other: the roadmap (the
// course as a path of lessons, and free play) and the keyboard. John, the singing head, sits large and half
// see-through behind both.

const NAMES = ["C", "C#", "D", "D#", "E", "F", "F#", "G", "G#", "A", "A#", "B"];
// Computer keys for one octave and a bit from C4, like a tracker: A W S E D F T G Y H U J K.
const COMPUTER = { a: 0, w: 1, s: 2, e: 3, d: 4, f: 5, t: 6, g: 7, y: 8, h: 9, u: 10, j: 11, k: 12, o: 13, l: 14 };
const MODE_WORD = { listen: "Listen", learn: "Learn", watch: "Repeat", together: "Together", jam: "Jam" };

export const isBlack = (midi) => NAMES[midi % 12].endsWith("#");
export const noteName = (midi) => `${NAMES[midi % 12]}${Math.floor(midi / 12) - 1}`;

const $ = (id) => document.getElementById(id);
let state = null;
let drawnRange = "";
let drawnSongs = "";
let drawnCourse = "";
let lastPress = 0;
const judged = new Set();
let lastJohn = 0;
let holdPlay = 0; // keep the keyboard on screen until this time (ms), so a result can be read
let cheerTimer = 0;

// ---------------------------------------------------------------- server

function send(body) {
  return fetch("/cmd", { method: "POST", headers: { "Content-Type": "application/json" }, body: JSON.stringify(body) }).catch(() => {});
}

function connect() {
  const events = new EventSource("/events");
  events.addEventListener("state", (event) => render(JSON.parse(event.data)));
}

// ---------------------------------------------------------------- sound

let audio = null;
function ensureAudio() {
  if (!audio) audio = new AudioContext();
  if (audio.state === "suspended") audio.resume();
  $("chip-sound").hidden = true;
  return audio;
}

/** A plucked, piano-ish tone: a few harmonics with a fast attack and a long decay. */
function tone(midi, seconds = 1.2) {
  if (!audio) return; // no sound before the first click: the browser would refuse it anyway
  const hz = 440 * 2 ** ((midi - 69) / 12);
  const out = audio.createGain();
  const t = audio.currentTime;
  out.gain.setValueAtTime(0.0001, t);
  out.gain.exponentialRampToValueAtTime(0.22, t + 0.005);
  out.gain.exponentialRampToValueAtTime(0.0001, t + seconds);
  out.connect(audio.destination);
  [1, 0.45, 0.25, 0.12].forEach((level, i) => {
    const osc = audio.createOscillator();
    const gain = audio.createGain();
    osc.frequency.value = hz * (i + 1);
    gain.gain.value = level;
    osc.connect(gain).connect(out);
    osc.start(t);
    osc.stop(t + seconds);
  });
}

// ---------------------------------------------------------------- the keyboard

function buildKeys(lo, hi) {
  const box = $("keys");
  box.innerHTML = "";
  const whites = [];
  for (let m = lo; m <= hi; m++) if (!isBlack(m)) whites.push(m);
  const width = 100 / whites.length;
  for (let m = lo; m <= hi; m++) {
    const key = document.createElement("div");
    key.className = `key ${isBlack(m) ? "black" : "white"}`;
    key.dataset.midi = String(m);
    key.innerHTML = `<i class="reach"></i><span class="name">${noteName(m)}</span>`;
    if (isBlack(m)) {
      const left = whites.indexOf(m - 1);
      key.style.left = `calc(${(left + 1) * width}% - ${width * 0.3}%)`;
      key.style.width = `${width * 0.6}%`;
    }
    key.addEventListener("pointerdown", (event) => {
      event.preventDefault();
      ensureAudio();
      send({ do: "key", midi: m });
    });
    box.appendChild(key);
  }
}

const keyEl = (midi) => document.querySelector(`.key[data-midi="${midi}"]`);

function flash(midi, kind) {
  const el = keyEl(midi);
  if (!el) return;
  el.classList.remove("flash-ok", "flash-miss", "flash-arm");
  void el.offsetWidth; // restart the flash if it is already showing
  el.classList.add(`flash-${kind}`);
  setTimeout(() => el.classList.remove(`flash-${kind}`), kind === "miss" ? 700 : 420);
}

function placeArm(arm) {
  const mark = $("arm-mark");
  const el = arm.at == null ? null : keyEl(arm.at);
  document.querySelectorAll(".key.armat").forEach((k) => k.classList.remove("armat"));
  if (!el) {
    mark.classList.remove("on");
    return;
  }
  el.classList.add("armat");
  const box = $("keys").getBoundingClientRect();
  const r = el.getBoundingClientRect();
  mark.style.left = `${r.left - box.left + r.width / 2}px`;
  mark.textContent = arm.pressing != null ? "arm plays" : "arm";
  mark.classList.toggle("pressing", arm.pressing != null);
  mark.classList.add("on");
}

// ---------------------------------------------------------------- the roadmap

function renderSongs(next) {
  const key = JSON.stringify(next.songs);
  const select = $("song");
  if (key !== drawnSongs) {
    drawnSongs = key;
    const chosen = select.value;
    select.innerHTML = next.songs.map((s) => `<option value="${s.name}">${s.title}</option>`).join("");
    if (next.song && next.songs.some((s) => s.name === next.song.name)) select.value = next.song.name;
    else if (chosen) select.value = chosen;
  }
}

const TAG = { done: "Done", open: "Start", locked: "Locked" };

/** The line through the lesson circles: a dotted path for the whole road, a solid one up to where you are. */
function drawRoad(course) {
  const svg = $("road-line");
  const box = svg.getBoundingClientRect();
  const points = [...document.querySelectorAll("#road li button")].map((b) => {
    const r = b.getBoundingClientRect();
    return [r.left - box.left + r.width / 2, r.top - box.top + r.height / 2];
  });
  if (points.length < 2 || !box.width) return;
  const path = (pts) => pts.map(([x, y], i) => {
    if (i === 0) return `M${x},${y}`;
    const [px, py] = pts[i - 1];
    const mid = (px + x) / 2;
    return `C${mid},${py} ${mid},${y} ${x},${y}`;
  }).join(" ");
  $("road-path").setAttribute("d", path(points));
  const reached = Math.max(1, course.filter((l) => l.status === "done").length + 1);
  $("road-done").setAttribute("d", path(points.slice(0, Math.min(reached, points.length))));
}

/** The course, twice: the big road on the first screen, and small steps under the keyboard. */
function renderCourse(next) {
  const key = JSON.stringify([next.course, next.lesson]);
  if (key !== drawnCourse) {
    drawnCourse = key;
    const road = $("road");
    const steps = $("steps");
    road.innerHTML = "";
    steps.innerHTML = "";
    next.course.forEach((lesson, i) => {
      const item = document.createElement("li");
      item.className = lesson.status;
      const b = document.createElement("button");
      b.type = "button";
      b.disabled = lesson.status === "locked";
      b.textContent = lesson.status === "done" ? "\u2713" : String(i + 1);
      b.setAttribute("aria-label", `Lesson ${i + 1}, ${lesson.title}, ${lesson.status}`);
      b.addEventListener("click", () => {
        ensureAudio();
        send({ do: "lesson", id: lesson.id });
      });
      item.appendChild(b);
      item.insertAdjacentHTML("beforeend", `<span class="tag">${TAG[lesson.status]}</span><span class="title">${lesson.title}</span><span class="goal">${lesson.goal}</span>`);
      road.appendChild(item);

      const step = document.createElement("li");
      const done = i > 0 && next.course[i - 1].status === "done";
      step.className = `${lesson.status}${next.lesson === lesson.id ? " running" : ""}${done ? " after-done" : ""}`;
      step.innerHTML = `<span>${lesson.status === "done" ? "\u2713" : i + 1}</span>`;
      steps.appendChild(step);
    });
    requestAnimationFrame(() => drawRoad(next.course));
    const doneCount = next.course.filter((l) => l.status === "done").length;
    $("done-count").textContent = doneCount;
    $("lesson-count").textContent = next.course.length;
    const upcoming = next.course.find((l) => l.status === "open");
    $("home-small").textContent = upcoming
      ? `${doneCount ? "Next" : "Start with"} lesson ${next.course.indexOf(upcoming) + 1}: ${upcoming.title}. ${upcoming.goal}.`
      : "You finished the course. Replay any lesson, or play freely.";
  }
  // Hearts only show inside a lesson: free play costs nothing.
  const hearts = $("hearts");
  const before = hearts.dataset.left;
  hearts.innerHTML = next.hearts == null ? "" : [0, 1, 2].map((i) => `<i class="${i < next.hearts ? "" : "lost"}${String(i) === String(next.hearts) && before && Number(before) > next.hearts ? " pop" : ""}">\u2665\uFE0E</i>`).join("");
  hearts.dataset.left = next.hearts == null ? "" : String(next.hearts);
  $("xp").textContent = next.xp;
  $("days").textContent = next.days;
}

/** Which screen shows. The keyboard while something runs, and a moment longer so its last words can be read. */
function showView(was, next) {
  const now = Date.now();
  if (next.mode) holdPlay = now + 400;
  else if (was?.mode && !next.mode) {
    const lessonEnded = !!was.lesson;
    const cheer = next.big === "Lesson complete" || next.big === "Out of hearts";
    if (lessonEnded && cheer) {
      $("cheer-big").textContent = next.big;
      $("cheer-small").textContent = next.small;
      $("cheer").dataset.tone = next.big === "Out of hearts" ? "miss" : "ok";
      $("cheer").hidden = false;
      clearTimeout(cheerTimer);
      cheerTimer = setTimeout(() => ($("cheer").hidden = true), 3200);
      holdPlay = now + 600;
    } else holdPlay = now + 3500;
    setTimeout(() => state && showView(state, state), holdPlay - now + 50);
  }
  const view = next.mode || now < holdPlay ? "play" : "home";
  if (document.body.dataset.view !== view) {
    document.body.dataset.view = view;
    document.activeElement?.blur?.(); // a focused button of the screen that slides away must not pull the page
    for (const box of [document.scrollingElement, document.body, document.querySelector("main")]) box?.scrollTo?.(0, 0);
    if (view === "home") requestAnimationFrame(() => drawRoad(next.course));
  }
}

// ---------------------------------------------------------------- render

function render(next) {
  const was = state;
  state = next;
  const [lo, hi] = next.range;
  if (drawnRange !== `${lo}-${hi}`) {
    buildKeys(lo, hi);
    drawnRange = `${lo}-${hi}`;
  }
  renderSongs(next);
  renderCourse(next);
  showView(was, next);
  // Reachy's spoken guide: what it heard, then what it says.
  const voice = next.voice ?? { on: false };
  $("voice").hidden = !voice.on;
  if (voice.on) {
    const said = voice.reachy ? `<b>Reachy:</b> ${voice.reachy}` : voice.you ? `<b>You:</b> ${voice.you}` : 'Say "Hi Reachy", or "start lesson one".';
    if ($("voice-text").innerHTML !== said) $("voice-text").innerHTML = said;
  }

  $("chip-arm").textContent = next.arm.name || "Arm";
  $("chip-arm").classList.toggle("on", !!next.arm.name);
  $("chip-midi").classList.toggle("on", next.midi);
  $("chip-midi").textContent = next.midi ? "MIDI keyboard" : "No MIDI: click the keys";

  const lessonAt = next.course.findIndex((lesson) => lesson.id === next.lesson);
  if (lessonAt >= 0) $("eyebrow").textContent = `Lesson ${lessonAt + 1} · ${next.course[lessonAt].title}`;
  else $("eyebrow").textContent = next.mode ? `${MODE_WORD[next.mode]}${next.song ? ` · ${next.song.title}` : ""}` : "Keys101";
  $("big").textContent = next.big;
  $("big").dataset.tone = next.tone || "";
  $("small").textContent = next.small;

  $("phrase").innerHTML = next.notes.map((n) => `<span class="${n.status}">${n.name}</span>`).join("");
  const s = next.score;
  const where = next.phrases ? `phrase ${next.phrase + 1} of ${next.phrases} · ` : "";
  $("score").innerHTML = s.right + s.wrong ? `${where}<b>${s.right}</b> right · <b>${s.wrong}</b> ${s.wrong === 1 ? "slip" : "slips"}` : "";

  const reach = new Set(next.reach);
  const down = new Set(next.keys_down);
  document.querySelectorAll(".key").forEach((el) => {
    const m = Number(el.dataset.midi);
    el.classList.toggle("target", next.target === m);
    el.classList.toggle("down", down.has(m));
    el.classList.toggle("can", reach.has(m));
  });
  placeArm(next.arm);

  for (const p of next.presses) {
    if (p.id > lastPress) {
      const play = p.who === "ghost" || (p.who === "arm" && next.sound.arm) || (p.who === "you" && next.sound.you);
      if (play) tone(p.midi);
      if (p.who !== "you") flash(p.midi, "arm");
    }
    if (p.who === "you" && p.ok !== null && !judged.has(p.id)) {
      judged.add(p.id);
      flash(p.midi, p.ok ? "ok" : "miss");
    }
  }
  lastPress = Math.max(lastPress, ...next.presses.map((p) => p.id), 0);

  john.follow(was, next);
}

// ---------------------------------------------------------------- John (web/chimera, an iframe)

const john = (() => {
  const frame = $("john-frame");
  if (new URLSearchParams(location.search).get("chimera") === "0") {
    $("john").hidden = true;
    return { follow() {} };
  }
  frame.src = "chimera/teacher.html";
  let ready = false;
  let started = false;
  const queue = [];
  const post = (message) => {
    if (ready) frame.contentWindow?.postMessage(message, location.origin);
    else queue.push(message);
  };
  window.addEventListener("message", (event) => {
    if (event.origin !== location.origin || event.source !== frame.contentWindow) return;
    const data = event.data;
    if (!data || data.source !== "chimera-coach") return;
    if (data.type === "ready") {
      ready = true;
      while (queue.length) frame.contentWindow.postMessage(queue.shift(), location.origin);
    } else if (data.type === "started") started = true;
  });
  const nudge = () => {
    ensureAudio();
    if (!started) frame.contentWindow?.__coachNudge?.();
  };
  window.addEventListener("pointerdown", nudge, true);
  window.addEventListener("keydown", nudge, true);
  return {
    follow(was, now) {
      if (now.john && now.john.id !== lastJohn) {
        lastJohn = now.john.id;
        post({ type: "sing", midi: now.john.midi, seconds: now.john.seconds, name: now.john.name, backing: now.mode === "jam" });
      } else if (now.mode !== was?.mode) {
        const text = {
          listen: "The arm plays. I hum along.",
          learn: "Follow the arm. If you miss, I sing the right one.",
          watch: "Watch the arm, then play it back.",
          together: "You lead. The arm and I follow.",
          jam: "Jam time. I do backing vocals.",
          "": "Pick a lesson on the road. I'll warm up my vocal folds.",
        }[now.mode] ?? null;
        if (text) post({ type: "idle", text });
      }
    },
  };
})();

// ---------------------------------------------------------------- controls

document.querySelectorAll(".modes button[data-mode]").forEach((b) => {
  b.addEventListener("click", () => {
    ensureAudio();
    send({ do: "start", mode: b.dataset.mode, song: $("song").value });
  });
});
$("make-up").addEventListener("click", () => {
  ensureAudio();
  send({ do: "make_up", mode: "learn" });
});
$("stop").addEventListener("click", () => send({ do: "stop" }));
$("reset").addEventListener("click", () => {
  if (confirm("Start the course over? Your XP and finished lessons are cleared.")) send({ do: "reset_course" });
});
window.addEventListener("keydown", (event) => {
  if (event.repeat || event.metaKey || event.ctrlKey || event.target.tagName === "SELECT") return;
  const step = COMPUTER[event.key.toLowerCase()];
  if (step === undefined || !state) return;
  const midi = 60 + step;
  if (midi < state.range[0] || midi > state.range[1]) return;
  ensureAudio();
  send({ do: "key", midi });
});
window.addEventListener("resize", () => {
  if (!state) return;
  placeArm(state.arm);
  drawRoad(state.course);
});

// The page never scrolls: if a focused control of the screen parked off to the side pulls it, put it back.
const unscroll = () => {
  for (const box of [document.scrollingElement, document.body, document.querySelector("main")]) {
    if (box && (box.scrollLeft || box.scrollTop)) box.scrollTo(0, 0);
  }
};
window.addEventListener("scroll", unscroll, true);

connect();
