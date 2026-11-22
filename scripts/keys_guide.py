#!/usr/bin/env python
"""Reachy talks: the spoken guide of Keys101 (keys_coach.py --voice).

You speak to Reachy Mini and it answers through its own speaker. It introduces the game, and it starts what
you ask for: "start lesson one", "next lesson", "let's jam", "play Ode to Joy together", "make up a
tune", "stop".

Three parts, each doing one job:
  GPT-Live     listens and speaks (OpenAI, over a WebSocket). It cannot move anything: every request to
               start or stop something is handed to this script ("delegation").
  Voice Focus  Quail VF 2.2 at full strength cleans your voice on its way there: the room, the piano and
               other people are turned down. Off when the ai-coustics SDK or its key is missing.
  the coach    starts the lesson or the mode, and when it ends Reachy says how it went.

You need no magic words. Reachy works out what you want from the whole conversation, asks when it is
not sure, and announces what it starts ("Lesson one, here we go!"); this script starts the one thing that
sentence names (decide()). A question or two options start nothing, so you can just talk.

GPT-Live only speaks in reply to you, never by itself: say hello first. While the arm plays or you play,
its microphone is closed, so it does not answer the piano. Its head is keys_head.py's job.

Needs OPENAI_API_KEY (the environment, else .env in the repository root). Run keys_coach.py under
    uv run --with-requirements scripts/requirements-keys.txt --with aic-sdk==3.3.2
(Quail VF 2.2 needs that SDK version: 3.4.0 only loads VF 2.3.)
"""

from __future__ import annotations

import asyncio
import contextlib
import json
import re
import threading
import time
from typing import Any

import numpy as np

URL = "wss://api.openai.com/v1/live/sessions"
MODELS = ("gpt-live-1.1", "gpt-live-1")  # tried in this order
VOICE_FOCUS_MODEL = "quail-vf-2.2-l-16khz"
ENHANCEMENT = 1.0  # Voice Focus at full strength
RATE = 16000
MIC = "MacBook Pro Microphone"

NUMBER = {"1": 1, "2": 2, "3": 3, "4": 4, "5": 5, "6": 6, "7": 7, "one": 1, "first": 1, "two": 2, "second": 2, "three": 3, "third": 3, "four": 4, "fourth": 4,
          "five": 5, "fifth": 5, "six": 6, "sixth": 6, "seven": 7, "seventh": 7, "last": -1}  # fmt: skip
SONG_WORDS = {"mary": ("mary", "lamb"), "ode": ("ode", "joy", "beethoven"), "twinkle": ("twinkle", "star"),
              "canon": ("canon", "pachelbel"), "fur_elise": ("elise", "elisa"),
              "alors": ("alors", "danse", "dance", "stromae")}  # fmt: skip
MODE_WORDS = {
    "jam": ("jam", "jamming", "improvise", "answer me", "call and response"),
    "together": ("together", "duet", "with me", "bass", "band", "accompany"),
    "listen": ("listen", "for me", "yourself", "itself", "on its own", "alone", "solo", "perform"),
    "watch": ("repeat", "play it back", "watch", "show me"),
    "sing": ("sing", "singing", "my voice", "pitch", "ear training"),
    "learn": ("learn", "learning", "teach", "practise", "practice"),
}
TUNE_WORDS = ("make up", "made up", "making up", "invent", "compose", "new song", "new tune", "new melody",
              "your own", "my own", "surprise", "something new", "come up with", "on the spot")  # fmt: skip
STOP_WORDS = ("stop", "stopping", "stopped", "enough", "go back", "roadmap", "quit", "cancel")
GO_WORDS = ("start", "begin", "next", "continue", "lesson")


def instructions(course: list[dict[str, Any]], songs: dict[str, Any]) -> str:
    lessons = ", ".join(f"{i}: {lesson['title']}" for i, lesson in enumerate(course, start=1))
    titles = ", ".join(song.title for song in songs.values())
    return f"""You are Reachy, a small robot with two antennas: curious, quick, a little cheeky, proud of \
your piano skills and always kind. You and your friend the robot arm run Keys101: the arm teaches a person \
a song on a small piano keyboard, and then plays in a band with them.

This is a spoken conversation. Talk like a person: one or two short sentences, react to what they actually \
said, vary your words, never say a sentence you already said, never recite a list of commands or of \
songs: name at most two songs at a time. English only.

What the two of you can do:
- Lessons on the roadmap, which unlock in order: {lessons}.
- Listen: the arm plays a whole song by itself while they watch. Songs: {titles}.
- Learn a song: the arm points at each key and waits for them.
- Repeat: the arm plays a phrase and they play it back.
- Together: they play the tune and the arm is their bass player. This is the band.
- Jam: they play any short phrase and the arm answers it.
- Sing: the arm plays a note, they sing it back, and a pitch tracker says how many cents off they were.
- A tune you make up on the spot, which the arm then teaches.

Work out what they want. They should never need magic words.
- New to the piano, or unsure: lesson one, only three keys.
- "I want to learn": ask once whether they have played before, then suggest something that fits.
- Has played before: Ode to Joy, or playing together.
- Wants a band, a duet, a bass player, company: together.
- Wants to improvise or show off: a jam.
- Wants to sing, train their ear or check their pitch: sing, with a song they know.
- Bored, wants a surprise or something new: a tune you make up.
- Wants to hear something, "play something for me", "show me what you can do": the arm plays a song \
by itself. Offer a song or two; if they say you choose, choose one and play it.

You cannot move the arm yourself. When they want something to start, change or stop, or they say yes to \
a suggestion of yours, hand it over. You then get a private note. Never read a note aloud and never quote \
it. After the note, do exactly one of two things:
1. It is clear what to start: announce it in one short, sure sentence that names exactly one thing (a \
song by its title), for example \
"Lesson one, here we go!", "Sit back, the arm plays Ode to Joy on its own!", "Ode to Joy together, I \
bring the bass!", "Let's sing the notes of Mary Had a Little Lamb!", "A jam it is!", "I'll make up a tune and teach it to you!". No question and no second option in that sentence. It starts the moment you finish, so then \
you stay quiet.
2. It is not clear yet: ask or suggest, in a sentence with a question mark.

When something ends you get a note about how it went. React like a friend would, in your own words, and \
say what could come next."""


def brief(course: list[dict[str, Any]]) -> str:
    """The private note Reachy gets when it hands a request over: where the course stands, and how to
    answer so that this script can follow (decide())."""
    done = sum(lesson["status"] == "done" for lesson in course)
    upcoming = next((i for i, lesson in enumerate(course, start=1) if lesson["status"] == "open"), None)
    where = (
        f"{done} of {len(course)} lessons are done; lesson {upcoming} ({course[upcoming - 1]['title']}) "
        f"is the next one, later lessons are locked."
        if upcoming
        else "Every lesson is done; any can be replayed."
    )
    return (
        f"Private note, never read it aloud. {where} If you know what they want, announce it now in one "
        "short sentence naming exactly one thing: a lesson by its number, the next lesson, a song the "
        "arm plays on its own, a song to learn, repeat, sing or play together, a jam, a tune you make up, or "
        "stopping. Otherwise ask them, "
        "with a question mark."
    )


# An offer is not a start: "I can teach you a song if you're interested" must not begin a lesson.
HEDGES = ("if you", "i can ", "i could", "we can ", "we could", "you can ", "you could", "would you", "do you",
          "interested", "maybe", "perhaps", "don't", "can't", "cannot", "not sure", "how about",
          "want to", "wanna", "like to", "sorry")  # fmt: skip


def say_exactly(line: str) -> str:
    """A note that makes Reachy say one line, word for word: the singing lesson's cues and verdicts."""
    return f'Private note. Say exactly these words and nothing else: "{line}"'


def decide(announced: str, course: list[dict[str, Any]], songs: dict[str, Any]) -> tuple[str, ...]:
    """What to start, from Reachy's own announcement. It has the whole conversation in its head, so it
    does the inferring; this only reads the one thing it named. A question, an offer, two options, or a
    song mode without its song start nothing: the conversation simply goes on."""
    text = " ".join(announced.lower().replace("\u2019", "'").split())
    if not text or "?" in text or re.search(r"\bor\b", text) or any(h in text + " " for h in HEDGES):
        return ("talk",)
    action = parse(text, course, songs)
    if action[0] == "mode" and action[1] != "jam":
        words = re.sub(r"[^a-z0-9' ]", " ", text).split()
        named = [n for n, keys in SONG_WORDS.items() if n in songs and any(k in words for k in keys)]
        if len(named) != 1:
            return ("talk",)  # "let's learn a song": which one?
    return ("talk",) if action == ("unknown",) else action


def parse(text: str, course: list[dict[str, Any]], songs: dict[str, Any]) -> tuple[str, ...]:
    """What a spoken request means: ("stop",), ("lesson", id), ("mode", mode, song), ("tune", mode) or
    ("unknown",).
    `course` is the roadmap as the page has it: each lesson with its status."""
    said = " ".join(re.sub(r"[^a-z0-9' ]", " ", text.lower()).split())
    words = said.split()

    def has(keys: tuple[str, ...]) -> bool:  # whole words: "start" is not Twinkle's "star"
        return any(re.search(rf"\b{re.escape(k)}", said) if " " in k else k in words for k in keys)

    if has(STOP_WORDS):
        return ("stop",)
    song = next((name for name, keys in SONG_WORDS.items() if name in songs and has(keys)), None)
    mode = next((m for m, keys in MODE_WORDS.items() if has(keys)), None)
    if has(TUNE_WORDS):
        return ("tune", mode if mode in ("listen", "watch", "together", "sing") else "learn")
    number = next((NUMBER[w] for w in words if w in NUMBER), None)
    if number is not None and "lesson" in words:
        index = len(course) - 1 if number == -1 else number - 1
        if 0 <= index < len(course):
            return ("lesson", course[index]["id"])
    if mode == "jam":
        return ("mode", "jam", "")
    if mode or song:
        return ("mode", mode or "learn", song or next(iter(songs)))
    if any(w in words for w in GO_WORDS) or "lesson" in words:
        upcoming = next((lesson for lesson in course if lesson["status"] == "open"), None)
        if upcoming is not None:
            return ("lesson", upcoming["id"])
    return ("unknown",)


class VoiceFocus:
    """Quail Voice Focus 2.2 on a live stream, at full enhancement: your voice, with the room turned down."""

    def __init__(self) -> None:
        import audio_hooks

        aic = audio_hooks._sdk()
        model = audio_hooks._load_model(aic, VOICE_FOCUS_MODEL)
        config = aic.ProcessorConfig.optimal(model, sample_rate=RATE)
        self.processor = aic.Processor(model, audio_hooks.license_key())
        self.processor.initialize(config)
        self.processor.get_context().set_parameter(aic.ProcessorParameter.EnhancementLevel, ENHANCEMENT)
        self.block = int(config.block_size)
        self.pending = np.zeros(0, dtype=np.float32)

    def __call__(self, audio: np.ndarray) -> np.ndarray:
        self.pending = np.concatenate([self.pending, audio.astype(np.float32)])
        out = []
        while self.pending.size >= self.block:
            out.append(
                np.asarray(self.processor.process(self.pending[: self.block].copy()), dtype=np.float32)
            )
            self.pending = self.pending[self.block :]
        return np.concatenate(out) if out else np.zeros(0, dtype=np.float32)


class Guide:
    """The spoken guide, in its own thread. It never blocks the coach: if anything fails it prints why once
    and the game goes on by clicks."""

    def __init__(
        self,
        coach: Any,
        state: Any,
        songs: dict[str, Any],
        mic: str | None = MIC,
        cartoon: float = 1.0,
        speaker: str | None = None,
    ) -> None:
        self.speaker = speaker  # where Reachy's voice comes out (None: Reachy's own speaker, else the Mac)
        self.coach, self.state, self.songs, self.mic = coach, state, songs, mic
        self.cartoon = cartoon  # the voice this much faster and higher
        self.busy = False  # something is playing: the microphone is closed
        self.heard = ""  # what you said since Reachy last spoke
        self.spoken = ""  # Reachy's words on the screen: its last turn
        self.said = ""  # everything Reachy said, for reading its announcements
        self.said_at = 0.0
        self.pending: str | None = None  # the delegation waiting to hear how what it started went
        self.last: str | None = None  # the newest delegation: Reachy can be given a line to say on it
        self._loop: asyncio.AbstractEventLoop | None = None
        self._say: Any = None  # say a line now (a coroutine function), while a conversation is open
        self.stop = threading.Event()
        self._thread = threading.Thread(target=self._run, daemon=True)

    def start(self) -> None:
        self._thread.start()

    def close(self) -> None:
        self.stop.set()

    def speak(self, line: str, timeout: float = 12.0) -> bool:
        """Reachy says this line, word for word, and this returns once it has. For other threads (the coach,
        in the middle of a lesson). False when it cannot speak now: no conversation has started yet."""
        if self._loop is None or self._say is None or (self.pending or self.last) is None:
            return False
        try:
            return bool(asyncio.run_coroutine_threadsafe(self._say(line), self._loop).result(timeout))
        except Exception:  # noqa: BLE001  a lesson never waits on a voice that fails
            return False

    def _run(self) -> None:
        try:
            asyncio.run(self._main())
        except BaseException as error:  # noqa: BLE001  SystemExit too: a missing key must not end the game
            print(f"[Reachy's voice] off: {str(error)[:300]}", flush=True)
            self.state.set(voice={"on": False, "you": "", "reachy": ""})

    async def _main(self) -> None:
        import keys_voice as kv
        from websockets.asyncio.client import connect

        key = kv.api_key()
        loop = asyncio.get_running_loop()
        self._loop = loop
        speaker = self.speaker or kv.reachy_speaker()
        audio = kv.Audio(self.mic, speaker, loop)
        clean: Any = None
        try:
            clean = VoiceFocus()
            focus = f"Voice Focus {VOICE_FOCUS_MODEL} at {ENHANCEMENT:.0%}"
        except BaseException as error:  # noqa: BLE001  the SDK exits when it or its key is missing
            focus = f"Voice Focus off ({str(error)[:80]})"
        problem = "no model answered"
        try:
            for model in MODELS:
                try:
                    async with connect(
                        URL, additional_headers={"Authorization": f"Bearer {key}"}, max_size=None
                    ) as socket:
                        print(
                            f"[Reachy's voice] {model}, speaker {speaker or 'the Mac'}, "
                            f"microphone {self.mic or 'the default'}, {focus}",
                            flush=True,
                        )
                        if await self._session(socket, model, audio, clean, kv):
                            return
                except Exception as error:  # noqa: BLE001  try the next model
                    problem = f"{model}: {str(error)[:200]}"
                    print(f"[Reachy's voice] {problem}", flush=True)
            raise RuntimeError(problem)
        finally:
            audio.close()

    async def _session(self, socket: Any, model: str, audio: Any, clean: Any, kv: Any) -> bool:
        """One conversation with one model. True when it ended because the app is closing."""

        async def send(event: dict[str, Any]) -> None:
            await socket.send(json.dumps(event))

        async def tell(delegation_id: str, content: str) -> None:
            print(f"[to Reachy] {content}", flush=True)
            await send(
                {"type": "session.commentary.append", "delegation_id": delegation_id, "content": content}
            )

        await send(
            {
                "type": "session.start",
                "session": {
                    "model": model,
                    "instructions": instructions(self.state.get("course"), self.songs),
                    "audio": {"format": {"type": "audio/pcm", "rate": RATE}, "output": {"voice": "meridian"}},
                    "delegation": {"type": "client"},
                },
            }
        )
        started = asyncio.Event()
        refused: list[str] = []
        done = asyncio.Event()
        cartoon = kv.Speedup(self.cartoon)

        def show() -> None:
            self.state.set(voice={"on": True, "you": self.heard[-160:], "reachy": self.spoken[-220:]})

        async def announcement(mark: int, quiet: float = 1.4) -> str:
            """What Reachy says after a note: wait until it has spoken and gone quiet again."""
            deadline = time.monotonic() + 12.0
            while time.monotonic() < deadline and not done.is_set():
                await asyncio.sleep(0.1)
                if len(self.said) > mark and time.monotonic() - self.said_at > quiet:
                    break
            await audio.finished_speaking()
            return self.said[mark:]

        async def say_now(line: str) -> bool:
            target = self.pending or self.last
            if target is None or done.is_set():
                return False
            mark = len(self.said)
            await tell(target, say_exactly(line))
            await announcement(mark, quiet=0.7)
            return True

        self._say = say_now

        async def handle(delegation_id: str) -> None:
            self.last = delegation_id
            request, self.heard = self.heard, ""
            course = self.state.get("course")
            if parse(request, course, self.songs) == ("stop",):  # stopping never waits for a sentence
                print(f"\n[you] {request.strip()!r} -> stop", flush=True)
                await asyncio.to_thread(self.coach.stop)
                await tell(delegation_id, "Private note: it is stopped, the roadmap is back. Ask what next.")
                return
            mark = len(self.said)
            await tell(delegation_id, brief(course))
            said = await announcement(mark)
            action = decide(said, course, self.songs)
            print(f"\n[you] {request.strip()!r}  [Reachy] {said.strip()!r} -> {action}", flush=True)
            if action[0] == "stop":
                await asyncio.to_thread(self.coach.stop)
            elif action[0] == "lesson":
                index = next(i for i, lesson in enumerate(course) if lesson["id"] == action[1])
                if course[index]["status"] == "locked":
                    await tell(
                        delegation_id,
                        f"Private note: lesson {index + 1} is still locked, nothing started. Tell them "
                        "which lesson comes first, and ask if they want that one.",
                    )
                    return
                self.pending = delegation_id
                await asyncio.to_thread(self.coach.start_lesson, action[1])
            elif action[0] == "tune":
                self.pending = delegation_id
                await asyncio.to_thread(self.coach.make_up, action[1])
            elif action[0] == "mode":
                self.pending = delegation_id
                await asyncio.to_thread(self.coach.start, action[1], action[2] or None)

        tasks: set[asyncio.Task[None]] = set()

        async def receive() -> None:
            async for message in socket:
                event = json.loads(message)
                kind = event.get("type", "")
                if kind == "session.started":
                    started.set()
                elif kind == "session.output_audio.delta":
                    audio.say(cartoon(kv.from_pcm(event["delta"])))
                elif kind == "session.input_transcript.delta":
                    if self.spoken:
                        self.spoken = ""  # a new turn of yours: Reachy's last words leave the screen
                        self.heard = ""  # and what you said before it answered is over
                    self.heard += event.get("delta", "")
                    show()
                elif kind == "session.output_transcript.delta":
                    self.spoken += event.get("delta", "")
                    self.said += event.get("delta", "")
                    self.said_at = time.monotonic()
                    print(event.get("delta", ""), end="", flush=True)
                    show()
                elif kind == "session.delegation.created":
                    task = asyncio.create_task(
                        kv.guard(handle(event.get("delegation", {}).get("id")), "request")
                    )
                    tasks.add(task)
                    task.add_done_callback(tasks.discard)
                elif kind == "error":
                    message = json.dumps(event.get("error", event))[:300]
                    if not started.is_set():
                        refused.append(message)
                        started.set()
                    else:
                        print(f"\n[GPT-Live error] {message}", flush=True)
                elif kind == "session.closed":
                    break
            done.set()

        async def microphone() -> None:
            silence = np.zeros(kv.BLOCK, dtype=np.float32)
            while not done.is_set():
                block = await audio.blocks.get()
                if self.busy:
                    continue  # the server is muted: nothing to send
                if audio.speaking:
                    block = silence  # half duplex: Reachy would otherwise hear itself
                elif clean is not None:
                    block = clean(block)
                if block.size:
                    await send({"type": "session.input_audio.append", "audio": kv.to_pcm(block)})

        async def watch() -> None:
            """Close the microphone while anything plays, and tell Reachy how it went afterwards."""
            while not done.is_set():
                playing = bool(self.state.get("mode"))
                if playing and not self.busy:
                    self.busy = True
                    await send({"type": "session.input_audio.mute"})
                elif not playing and self.busy:
                    await asyncio.sleep(1.0)  # the piano's tail
                    self.busy = False
                    self.heard = ""
                    await send({"type": "session.input_audio.unmute"})
                    if self.pending:
                        delegation_id, self.pending = self.pending, None
                        big, small = self.state.get("big"), self.state.get("small")
                        how = (
                            "they stopped it before the end"
                            if big == "Stopped"
                            else f"it ended like this: {big}. {small}"
                        )
                        await tell(
                            delegation_id,
                            f"Private note, never read it aloud: {how} React to that in your own words, "
                            "one or two short sentences, and say what could come next.",
                        )
                if self.stop.is_set():
                    with contextlib.suppress(Exception):
                        await send({"type": "session.close"})
                    done.set()
                await asyncio.sleep(0.2)

        jobs = [asyncio.create_task(kv.guard(receive(), "receiving from GPT-Live", done))]
        try:
            await asyncio.wait_for(started.wait(), timeout=12)
            if refused:
                raise RuntimeError(refused[0])
            show()
            print('[Reachy\'s voice] listening. Say "Hi Reachy", or "start lesson one".', flush=True)
            jobs.append(asyncio.create_task(kv.guard(microphone(), "microphone", done)))
            jobs.append(asyncio.create_task(kv.guard(watch(), "watching the game", done)))
            await done.wait()
            return self.stop.is_set()
        finally:
            for job in [*jobs, *tasks]:
                job.cancel()


if __name__ == "__main__":  # a dry check of the request parser
    import sys

    demo_course = [
        {"id": f"l{i}", "title": f"Lesson {i}", "goal": "", "status": "open" if i == 1 else "locked"}
        for i in range(1, 8)
    ]
    demo_songs = dict.fromkeys(("mary", "ode", "twinkle", "canon", "fur_elise"))
    print(parse(" ".join(sys.argv[1:]) or "start lesson one", demo_course, demo_songs))
    time.sleep(0)
