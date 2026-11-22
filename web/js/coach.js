// Keys101: the page of scripts/keys_coach.py. It shows the coach's state and sends your choices back.
// Everything it shows comes from the server as one JSON state (GET /events). It sends: a lesson or a mode to
// start, stop, a key you clicked (POST /cmd). One screen: the words, the keyboard, and a dock with the course
// on the left and free play on the right.

const NAMES = ["C", "C#", "D", "D#", "E", "F", "F#", "G", "G#", "A", "A#", "B"];
// Computer keys for one octave and a bit from C4, like a tracker: A W S E D F T G Y H U J K.
const COMPUTER = { a: 0, w: 1, s: 2, e: 3, d: 4, f: 5, t: 6, g: 7, y: 8, h: 9, u: 10, j: 11, k: 12, o: 13, l: 14 };
const MODE_WORD = { learn: "Learn", watch: "Repeat", together: "Together", jam: "Jam" };

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

// ---------------------------------------------------------------- the dock

function renderSongs(next) {
  const key = JSON.stringify(next.songs);
  const select = $("song");
  if (key !== drawnSongs) {
    drawnSongs = key;
    const chosen = select.value;
    select.innerHTML = next.songs.map((s) => `<option value="${s.name}">${s.title}</option>`).join("");
    if (chosen) select.value = chosen;
  }
  if (next.song && next.mode && !next.lesson) select.value = next.song.name;
}

/** The course: a row of numbered steps. Done ones can be replayed, the next one is open, the rest are locked. */
function renderCourse(next) {
  const key = JSON.stringify([next.course, next.lesson]);
  if (key !== drawnCourse) {
    drawnCourse = key;
    const list = $("course");
    list.innerHTML = "";
    next.course.forEach((lesson, i) => {
      const item = document.createElement("li");
      if (i > 0 && next.course[i - 1].status === "done") item.className = "after-done";
      const b = document.createElement("button");
      b.type = "button";
      b.className = `${lesson.status}${next.lesson === lesson.id ? " running" : ""}`;
      b.disabled = lesson.status === "locked";
      b.textContent = lesson.status === "done" ? "✓" : String(i + 1);
      b.title = `${lesson.title}: ${lesson.goal}`;
      b.setAttribute("aria-label", `Lesson ${i + 1}, ${lesson.title}, ${lesson.status}`);
      b.addEventListener("click", () => {
        ensureAudio();
        send({ do: "lesson", id: lesson.id });
      });
      item.appendChild(b);
      list.appendChild(item);
    });
    const at = next.course.findIndex((l) => l.id === next.lesson);
    const upcoming = at >= 0 ? at : next.course.findIndex((l) => l.status === "open");
    const caption = $("caption");
    if (upcoming >= 0) {
      const lesson = next.course[upcoming];
      caption.innerHTML = `<b>${at >= 0 ? "Now" : "Next"}: ${lesson.title}.</b> ${lesson.goal}.`;
    } else {
      caption.innerHTML = next.course.length ? "<b>Course finished.</b> Replay any lesson, or play freely." : "";
    }
  }
  // Hearts only show inside a lesson: free play costs nothing.
  $("hearts").innerHTML = next.hearts == null ? "" : [0, 1, 2].map((i) => `<i class="${i < next.hearts ? "" : "lost"}">♥︎</i>`).join("");
  $("xp").textContent = next.xp;
  $("days").textContent = next.days;
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

  document.querySelectorAll(".modes button").forEach((b) => b.classList.toggle("on", b.dataset.mode === next.mode && !next.lesson));
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
          learn: "Follow the arm. If you miss, I sing the right one.",
          watch: "Watch the arm, then play it back.",
          together: "You lead. The arm and I follow.",
          jam: "Jam time. I do backing vocals.",
          "": "Pick a lesson. I'll warm up my vocal folds.",
        }[now.mode] ?? null;
        if (text) post({ type: "idle", text });
      }
    },
  };
})();

// ---------------------------------------------------------------- controls

document.querySelectorAll(".modes button").forEach((b) => {
  b.addEventListener("click", () => {
    ensureAudio();
    send({ do: "start", mode: b.dataset.mode, song: $("song").value });
  });
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
window.addEventListener("resize", () => state && placeArm(state.arm));

connect();
