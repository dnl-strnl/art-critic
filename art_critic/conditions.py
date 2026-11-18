"""Conditions: the creative angle a round of feedback is given under.

A condition is one value on one axis, e.g. (vibe, serene) or
(style, Impressionism). It has to be broad: something plenty of items in any
pool have more or less of, so that feedback under it is taste among items
that fit, not a test of whether anything fits. Labeled axes come from the
pool's labels, so which items fit is known. Text axes are matched to items
by CLIP similarity.
"""

from __future__ import annotations

from dataclasses import dataclass

CUSTOM_AXIS = "custom"

# axis -> {value: CLIP prompt}
TEXT_AXES: dict[str, dict[str, str]] = {
    "vibe": {
        "happy": "a happy, joyful artwork",
        "serene": "a serene, calm artwork",
        "melancholy": "a melancholy artwork",
        "eerie": "an eerie, unsettling artwork",
        "dreamlike": "a dreamlike artwork",
        "playful": "a playful, whimsical artwork",
        "dramatic": "a dramatic, intense artwork",
        "romantic": "a romantic, tender artwork",
        "nostalgic": "a nostalgic artwork",
        "mysterious": "a mysterious artwork",
        "energetic": "an energetic, lively artwork",
        "cozy": "a cozy, intimate artwork",
    },
    "aesthetic": {
        "minimal": "a minimal, sparse artwork",
        "ornate": "an ornate, decorative artwork",
        "geometric": "a geometric artwork",
        "gritty": "a gritty, raw artwork",
        "monumental": "a grand, monumental artwork",
        "surreal": "a surreal artwork",
        "vibrant": "a vibrant, colorful artwork",
        "moody": "a dark, moody artwork",
        "retro": "a retro artwork",
        "futuristic": "a futuristic artwork",
        "psychedelic": "a psychedelic artwork",
        "vaporwave": "a vaporwave aesthetic artwork",
    },
}

# Label axes too specific to judge taste under: a round on one is closer to a quiz.
SPECIFIC_LABEL_AXES = {"artist", "genre"}
# How a label value becomes a CLIP prompt; other label axes use the bare value.
LABEL_PROMPTS = {"style": "a {} painting"}


@dataclass(frozen=True)
class Condition:
    axis: str
    value: str
    prompt: str
    labeled: bool = False

    @property
    def key(self) -> str:
        return f"{self.axis}:{self.value}"

    @property
    def custom(self) -> bool:
        return self.axis == CUSTOM_AXIS

    def to_dict(self) -> dict:
        return {"key": self.key, "axis": self.axis, "value": self.value, "prompt": self.prompt}


def build(label_values: dict[str, list[str]], extra: dict[str, list[str]] | None = None) -> list[Condition]:
    """The condition vocabulary: label values, built-in text axes, and any extras.

    `extra` is {axis: [value, ...]}, e.g. modifiers pulled from your own
    prompts; each value doubles as its CLIP prompt.
    """
    out = [
        Condition(axis, value, LABEL_PROMPTS.get(axis, "{}").format(value), labeled=True)
        for axis, values in label_values.items()
        for value in values
    ]
    text = {axis: dict(values) for axis, values in TEXT_AXES.items()}
    for axis, values in (extra or {}).items():
        if axis in label_values or axis in (CUSTOM_AXIS, OPEN.axis):
            raise ValueError(f"extra condition axis {axis!r} collides with a reserved axis")
        text.setdefault(axis, {}).update({v: v for v in values})
    out += [Condition(axis, value, prompt) for axis, values in text.items() for value, prompt in values.items()]
    return out


def custom(text: str) -> Condition:
    return Condition(CUSTOM_AXIS, text, text)


# Feedback given with the condition switched off: plain taste, under no prompt.
OPEN = Condition("open", "no condition", "")
