"""Tests for Keys101 (keys_coach.py). No hardware, MIDI keyboard or browser needed.

uv run pytest scripts/test_coach.py -q
"""

from __future__ import annotations

import itertools
import json
import threading
import time
import urllib.request
from http.server import ThreadingHTTPServer

import keys_coach as kc
import pytest


def test_note_names_round_trip() -> None:
    assert kc.note_midi("C4") == 60
    assert kc.note_midi("F#3") == 54
    assert kc.note_midi("Bb3") == 58
    assert kc.note_midi("a4") == 69
    assert all(kc.note_midi(kc.midi_name(m)) == m for m in range(21, 109))
    for bad in ("H4", "C", "E#4", "C44"):
        with pytest.raises(ValueError):
            kc.note_midi(bad)


def test_every_lesson_loads_and_fits_the_keyboard() -> None:
    songs = kc.load_songs()
    assert {"ode", "mary", "twinkle", "canon", "fur_elise"} <= songs.keys()
    for song in songs.values():
        info = song.info()
        assert 48 <= info["low"] and info["high"] <= 72, f"{song.name} leaves C3 to C5"
        for phrase in song.phrases:
            for bar in phrase.bars:
                assert bar.arm is None or 48 <= bar.arm <= 72, song.name
    levels = [s.level for s in songs.values()]
    assert levels == sorted(levels)


def test_lesson_parsing() -> None:
    song = kc.parse_lesson("title: T\nyou: C4 D4 | E4\narm: C3 | -\n", "t")
    assert song.phrases[0].notes == [60, 62, 64]
    assert song.phrases[0].bar_starts() == {0: 48, 2: None}
    auto = kc.parse_lesson("you: C4 D4 | E4\n", "auto")  # no arm lines: first note an octave down
    assert auto.phrases[0].bar_starts() == {0: 48, 2: 52}
    for bad in (
        "arm: C3\n",
        "you: C4 | \n",
        "you: C4\narm: C3 | D3\n",
        "you: C4\narm: C3 D3\n",
        "tempo: 3\n",
    ):
        with pytest.raises(ValueError):
            kc.parse_lesson(bad, "bad")


def pose(x: float) -> kc.Pose:
    return {"shoulder_pan.pos": x, "shoulder_lift.pos": 2 * x, "gripper.pos": 10.0}


def board_of(*midis: int) -> kc.Board:
    return kc.Board(keys={m: {"above": pose(m), "down": pose(m + 0.5)} for m in midis})


def test_board_snaps_to_the_same_note_in_another_octave() -> None:
    board = board_of(60, 62, 64, 65, 67, 69)  # the six stickers
    assert board.snap(64) == 64
    assert board.snap(52) == 64  # E3 -> E4
    assert board.snap(48) == 60
    assert board.snap(71) is None  # no B anywhere
    assert board.nearest(71) == 69
    assert kc.Board(everything=True).snap(71) == 71


def test_fill_white_interpolates_untaught_keys() -> None:
    board = board_of(48, 60, 72)  # C3, C4, C5 taught; poses linear in the white-key index
    filled = kc.fill_white(board, 48, 72)
    assert len(filled) == 15 - 3
    assert not any(kc.is_black(m) for m in filled)
    whites = kc.white_keys(48, 72)
    # Taught poses are 48, 60, 72 at indices 0, 7, 14: the parabola through them is the line 48 + 12/7 * i.
    for m in filled:
        i = whites.index(m)
        assert board.keys[m]["above"]["shoulder_pan.pos"] == pytest.approx(48 + 12 / 7 * i, abs=1e-3)


def test_board_file_round_trip(tmp_path) -> None:
    board = board_of(60, 61)
    board.home = pose(1)
    kc.save_board(board, tmp_path / "b.json")
    again = kc.load_board(tmp_path / "b.json", tmp_path / "none.json")
    assert again.reach == [60, 61] and again.home == pose(1)


def test_board_falls_back_to_the_sticker_poses(tmp_path) -> None:
    poses = {"home": pose(0), **{f"{k}_{i}": pose(i) for i in range(1, 7) for k in ("above", "down")}}
    (tmp_path / "p.json").write_text(json.dumps({"poses": poses}))
    board = kc.load_board(tmp_path / "none.json", tmp_path / "p.json")
    assert board.reach == [48, 50, 52, 53, 55, 57]  # the bass: C3 D3 E3 F3 G3 A3


def test_twelve_sticker_layout(tmp_path) -> None:
    import keys_scripted
    import numpy as np

    names = ["C3", "D3", "E3", "F3", "G3", "A3", "C4", "D4", "E4", "F4", "G4", "A4"]
    (tmp_path / "layout.json").write_text(json.dumps({"names": names}))
    assert keys_scripted.read_layout(tmp_path / "layout.json") == names
    stickers = kc.read_stickers(tmp_path / "layout.json")
    assert stickers[1] == 48 and stickers[7] == 60 and stickers[12] == 69
    assert kc.read_stickers(tmp_path / "none.json") == {1: 48, 2: 50, 3: 52, 4: 53, 5: 55, 6: 57}
    # The repository's own layout is the twelve Keys101 keys, and every bass note of every lesson is in it.
    assert keys_scripted.read_layout() == names
    learned = set(kc.read_stickers().values())
    for song in kc.load_songs().values():
        for phrase in song.phrases:
            for bar in phrase.bars:
                assert bar.arm is None or bar.arm in learned or song.name == "fur_elise", song.name

    # Poses for twelve stickers become a twelve-key board.
    poses = {f"{k}_{i}": pose(i) for i in range(1, 13) for k in ("above", "down")}
    (tmp_path / "p12.json").write_text(json.dumps({"poses": poses}))
    board = kc.load_board(tmp_path / "none.json", tmp_path / "p12.json", stickers)
    assert board.reach == sorted(stickers.values())

    # A tour makes every jump between two keys, and every repeat, exactly once.
    keys = tuple(range(1, 13))
    tour = keys_scripted.all_pairs_tour(keys, np.random.default_rng(0))
    assert len(tour) == 12 * 12 + 1 and tour[0] == tour[-1] == 1
    assert set(itertools.pairwise(tour)) == {(a, b) for a in keys for b in keys}
    sequence = keys_scripted.tour_sequence(keys, 600, np.random.default_rng(1))
    counts = [sequence.count(k) for k in keys]
    assert len(sequence) == 600 and min(counts) >= 45 and max(counts) <= 55


def test_policy_prompts_match_the_recording() -> None:
    arm = kc.PolicyArm.__new__(kc.PolicyArm)
    arm.prompts = "notes"
    assert arm.task(52) == "press E3"
    arm.prompts = "scripted"
    assert arm.task(52) == "press key 3"


def test_answers() -> None:
    board = kc.Board(everything=True)
    assert kc.answer_phrase([60, 64, 67], "echo", board) == [60, 64, 67]
    assert kc.answer_phrase([60, 64, 67], "mirror", board) == [60, 56, 53]
    assert len(kc.answer_phrase(list(range(60, 72)), "echo", board)) == 6
    six = board_of(60, 62, 64, 65, 67, 69)
    assert kc.answer_phrase([72, 76], "echo", six) == [60, 64]


def test_ears_tell_the_arm_from_you() -> None:
    state = kc.State()
    ears = kc.Ears(state)
    ears.expect(64, 2.0)
    ears.key_down(64)  # the arm's press
    ears.key_down(64)  # the same key again: yours
    ears.key_down(60)
    got = [ears.queue.get_nowait() for _ in range(3)]
    assert [(p.midi, p.who) for p in got] == [(64, "arm"), (64, "you"), (60, "you")]
    assert [p["who"] for p in state.data["presses"]] == ["arm", "you", "you"]
    state.judge(got[1].id, False)
    assert state.data["presses"][1]["ok"] is False
    assert state.data["score"] == {"right": 0, "wrong": 1, "streak": 0}


# ---------------------------------------------------------------- the modes, with a simulated arm


def make_coach(**settings: float) -> tuple[kc.Coach, kc.Ears, kc.State]:
    state = kc.State()
    ears = kc.Ears(state)
    arm = kc.SimArm(ears, state, speed=50.0)
    songs = {"tiny": kc.parse_lesson("title: Tiny\nyou: C4 D4 | E4\narm: C3 | G3\n", "tiny")}
    coach = kc.Coach(arm, ears, state, songs, kc.Settings(**settings))
    return coach, ears, state


def wait_for(check, timeout: float = 5.0) -> None:
    end = time.monotonic() + timeout
    while time.monotonic() < end:
        if check():
            return
        time.sleep(0.01)
    raise AssertionError("timed out")


def play_when_asked(ears: kc.Ears, state: kc.State, wrong_first: int | None = None) -> threading.Thread:
    """A pupil who plays whatever key is lit, once wrong on the key `wrong_first`."""

    def run() -> None:
        last = None
        tricked = False
        end = time.monotonic() + 10
        while time.monotonic() < end and state.get("mode"):
            target = state.get("target")
            if target is not None and target != last:
                time.sleep(0.02)
                if target == wrong_first and not tricked:
                    tricked = True
                    ears.key_down(target + 1)
                    time.sleep(0.15)
                ears.key_down(target)
                last = target
            time.sleep(0.005)

    thread = threading.Thread(target=run, daemon=True)
    thread.start()
    return thread


def test_learn_mode_needs_two_clean_runs_and_replays_a_mistake() -> None:
    coach, ears, state = make_coach(passes=2, hesitate_s=0.05)
    coach.start("learn", "tiny")
    wait_for(lambda: state.get("mode") == "learn")
    play_when_asked(ears, state, wrong_first=62)
    wait_for(lambda: state.get("mode") == "", timeout=15)
    assert state.get("big").startswith("You can play Tiny")
    whos = [p["who"] for p in state.get("presses")]
    assert "arm" in whos  # the arm played D4 after the wrong key
    s = state.get("score")
    assert s["wrong"] == 1 and s["right"] >= 3 * 3  # one run with the slip, then two clean runs


def test_together_mode_plays_the_arm_note_on_each_bar() -> None:
    coach, ears, state = make_coach()
    coach.start("together", "tiny")
    wait_for(lambda: state.get("mode") == "together")
    play_when_asked(ears, state)
    wait_for(lambda: state.get("mode") == "", timeout=15)
    arm = [p["midi"] for p in state.get("presses") if p["who"] == "arm"]
    assert arm == [48, 55]


def test_jam_answers_and_stop_works() -> None:
    coach, ears, state = make_coach(phrase_end_s=0.1)
    coach.start("jam", None)
    wait_for(lambda: state.get("big") == "Your turn")
    for m in (60, 64, 67):
        ears.key_down(m)
    wait_for(lambda: len([p for p in state.get("presses") if p["who"] == "arm"]) == 3)
    coach.stop()
    assert state.get("mode") == ""


def test_server_serves_the_page_and_takes_commands() -> None:
    coach, ears, state = make_coach()
    server = ThreadingHTTPServer(("127.0.0.1", 0), kc.make_handler(coach, state, ears, kc.WEB.resolve()))
    threading.Thread(target=server.serve_forever, daemon=True).start()
    base = f"http://127.0.0.1:{server.server_address[1]}"
    try:
        page = urllib.request.urlopen(base + "/").read().decode()
        assert "Keys101" in page and "js/coach.js" in page
        assert json.loads(urllib.request.urlopen(base + "/state").read())["big"]
        for body in ({"do": "key", "midi": 60}, {"do": "style", "style": "mirror"}):
            req = urllib.request.Request(
                base + "/cmd", json.dumps(body).encode(), {"Content-Type": "application/json"}
            )
            urllib.request.urlopen(req).read()
        assert ears.queue.get(timeout=1).midi == 60
        assert state.get("style") == "mirror"
        with pytest.raises(urllib.error.HTTPError):
            urllib.request.urlopen(base + "/../keys_board.json")
    finally:
        server.shutdown()
