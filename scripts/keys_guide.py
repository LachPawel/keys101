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

NUMBER = {"one": 1, "first": 1, "two": 2, "second": 2, "three": 3, "third": 3, "four": 4, "fourth": 4,
          "five": 5, "fifth": 5, "six": 6, "sixth": 6, "seven": 7, "seventh": 7, "last": -1}  # fmt: skip
SONG_WORDS = {"mary": ("mary", "lamb"), "ode": ("ode", "joy", "beethoven"), "twinkle": ("twinkle", "star"),
              "canon": ("canon", "pachelbel"), "fur_elise": ("elise", "elisa")}  # fmt: skip
MODE_WORDS = {
    "jam": ("jam", "improvise", "answer me", "call and response"),
    "together": ("together", "duet", "with me", "bass", "accompany"),
    "watch": ("repeat", "play it back", "watch", "show me"),
    "learn": ("learn", "teach", "practise", "practice"),
}
TUNE_WORDS = ("make up", "made up", "invent", "compose", "new song", "new tune", "new melody", "your own",
              "surprise", "something new", "come up with", "on the spot")  # fmt: skip
STOP_WORDS = ("stop", "enough", "go back", "roadmap", "quit", "cancel")
GO_WORDS = ("start", "begin", "next", "continue", "go", "lesson", "ready", "let's", "yes", "okay")


def instructions(course: list[dict[str, Any]], songs: dict[str, Any]) -> str:
    lessons = ", ".join(f"{i}: {lesson['title']}" for i, lesson in enumerate(course, start=1))
    titles = ", ".join(song.title for song in songs.values())
    return f"""You are Reachy, a small friendly robot with two antennas. You run Keys101 together with a \
robot arm: the arm teaches the user a song on a small piano keyboard, then plays it with them.

What is on the screen: a roadmap of {len(course)} short lessons ({lessons}), and free play with four modes \
(learn, repeat, together, jam) and these songs: {titles}. In a lesson the arm hovers over each key and waits \
for the user; a wrong key costs one of three hearts and the arm plays the right one; a finished lesson gives \
points. In "together" the arm plays the bass while the user plays the tune. In "jam" the user plays a short \
phrase and the arm answers it. You can also make up a short new tune on the spot, and the arm teaches it: \
offer that when they want something new or a surprise.

When the user greets you or asks what this is: introduce the game in two or three short sentences, and \
invite them to say "start lesson one".

You cannot move the arm or change the screen yourself. Whenever the user wants to start, continue, choose \
or stop anything (a lesson, a mode, a song, "next", "go on", "stop"), hand the request over and wait for \
what you are told happened; then say that in one short sentence.

Always: one or two short sentences, warm and a little playful. Speak English. Never read out numbers of \
points or hearts unless you were told them."""


def parse(text: str, course: list[dict[str, Any]], songs: dict[str, Any]) -> tuple[str, ...]:
    """What a spoken request means: ("stop",), ("lesson", id), ("mode", mode, song), ("tune", mode) or
    ("unknown",).
    `course` is the roadmap as the page has it: each lesson with its status."""
    said = re.sub(r"[^a-z' ]", " ", text.lower())
    words = said.split()
    if any(w in said for w in STOP_WORDS):
        return ("stop",)
    song = next(
        (name for name, keys in SONG_WORDS.items() if name in songs and any(k in said for k in keys)), None
    )
    mode = next((m for m, keys in MODE_WORDS.items() if any(k in said for k in keys)), None)
    if any(w in said for w in TUNE_WORDS):
        return ("tune", mode if mode in ("watch", "together") else "learn")
    number = next((NUMBER[w] for w in words if w in NUMBER), None)
    names_a_lesson = "lesson" in words or (mode is None and song is None)
    if number is not None and names_a_lesson:
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
        self, coach: Any, state: Any, songs: dict[str, Any], mic: str | None = MIC, cartoon: float = 1.0
    ) -> None:
        self.coach, self.state, self.songs, self.mic = coach, state, songs, mic
        self.cartoon = cartoon  # the voice this much faster and higher
        self.busy = False  # something is playing: the microphone is closed
        self.heard = ""  # what you said since Reachy last spoke
        self.spoken = ""
        self.pending: str | None = None  # the delegation waiting to hear how what it started went
        self.stop = threading.Event()
        self._thread = threading.Thread(target=self._run, daemon=True)

    def start(self) -> None:
        self._thread.start()

    def close(self) -> None:
        self.stop.set()

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
        speaker = kv.reachy_speaker()
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

        async def handle(delegation_id: str) -> None:
            request, self.heard = self.heard, ""
            action = parse(request, self.state.get("course"), self.songs)
            print(f"\n[you] {request.strip()!r} -> {action}", flush=True)
            if action[0] == "stop":
                await asyncio.to_thread(self.coach.stop)
                await tell(delegation_id, "It is stopped and the roadmap is back on the screen.")
            elif action[0] == "lesson":
                course = self.state.get("course")
                index = next(i for i, lesson in enumerate(course) if lesson["id"] == action[1])
                lesson = course[index]
                if lesson["status"] == "locked":
                    await tell(
                        delegation_id, f"Lesson {index + 1} is still locked: the one before it comes first."
                    )
                    return
                self.pending = delegation_id
                await tell(
                    delegation_id,
                    f"Lesson {index + 1}, {lesson['title']}, starts now: {lesson['goal']}. Say that in one "
                    "short sentence, then be quiet while they play.",
                )
                await audio.finished_speaking()
                await asyncio.to_thread(self.coach.start_lesson, lesson["id"])
            elif action[0] == "tune":
                self.pending = delegation_id
                await tell(
                    delegation_id,
                    "You make up a short new tune right now and the arm teaches it. Say so in one short, "
                    "proud sentence, then be quiet while they play.",
                )
                await audio.finished_speaking()
                await asyncio.to_thread(self.coach.make_up, action[1])
            elif action[0] == "mode":
                mode, song = action[1], action[2]
                title = self.songs[song].title if song in self.songs else ""
                what = {
                    "learn": f"learning {title}: the arm shows each key",
                    "watch": f"{title}: the arm plays a phrase, they play it back",
                    "together": f"{title} together: they play the tune, the arm plays the bass",
                    "jam": "a jam: they play a short phrase, the arm answers",
                }[mode]
                self.pending = delegation_id
                await tell(delegation_id, f"Starting {what}. Say that in one short sentence, then be quiet.")
                await audio.finished_speaking()
                await asyncio.to_thread(self.coach.start, mode, song or None)
            else:
                await tell(
                    delegation_id,
                    "You did not catch what to start. Say they can ask for a lesson by its number, the next "
                    "lesson, free play (learn, repeat, together or jam), or a tune you make up.",
                )

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
                    self.heard += event.get("delta", "")
                    show()
                elif kind == "session.output_transcript.delta":
                    self.spoken += event.get("delta", "")
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
                        await tell(
                            delegation_id,
                            f"It is over. The screen says: {big}. {small} Say that warmly, in one or two short sentences.",
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
