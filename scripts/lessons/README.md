# Lessons for the piano coach

One song per file, read by `scripts/keys_coach.py`. All of them are public domain.

```
title: Ode to Joy
about: Beethoven, 1824. Five notes, right hand.
level: 1
you: E4 E4 F4 G4 | G4 F4 E4 D4
arm: C3          | G3
```

- `you:` one phrase of the part you learn. Notes with their octave (`C4`, `F#3`, `Bb3`), bars split by `|`.
  There is no rhythm: you lead, the coach waits for you.
- `arm:` optional, right under a `you:` line: one note per bar that the arm plays in "Play together", the
  moment you play the first note of that bar. `-` means no note in that bar. Without `arm:` lines the arm
  plays the first note of each bar an octave lower (none below C3).
- `#` at the start of a line or after a space starts a comment (`D#4` is a note).

The arm can only press keys it has poses for (`keys_board.json`). A note it cannot reach moves to the
nearest octave it can, else the screen plays it instead.
