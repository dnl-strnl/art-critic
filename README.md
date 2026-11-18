# art-critic

Teach a model your taste, one prompt at a time.

**A prompt sets the alignment task. Your taste picks the winner.** This is the RLHF loop for visual preference, conditioned on a prompt. Each round names a condition (a vibe, an aesthetic, a style, or anything you type), shows four images or videos that all fit it, and asks which you prefer. A reward model refits after every round.

Because everything in a round already fits the condition, your feedback is about preference, not about matching. The chart in the side panel (`\`) tracks exactly that: how often the model predicts your favorite, against how often plain prompt matching does.

Nothing is fine-tuned. The encoder ([CLIP ViT-B/32](https://huggingface.co/laion/CLIP-ViT-B-32-laion2B-s34B-b79K)) is frozen; the only thing trained is a small reward model on top of its embeddings.

## Run

```sh
uv sync
uv run python scripts/fetch_wikiart.py
uv run python -m art_critic  # http://127.0.0.1:8000
```

The first launch embeds everything (about a minute for 2,000 images on CPU); later launches only embed new files.

Your own media: `uv run python -m art_critic --images /path/to/folder`. Images and videos (`.mp4`, `.m4v`, `.mov`, `.webm`) can be mixed. State for each folder is kept under `data/state/`.

## Conditions

A condition is one value on one axis. It is meant to be broad, like "happy" or "interesting": something plenty of items have more or less of, so a round measures your taste among them. A narrow one, like a single artist, turns the round into a quiz on whether anything fits.

There are three sources:

- **Built-in text axes.** `vibe` (happy, serene, eerie, nostalgic, ...) and `aesthetic` (minimal, ornate, surreal, vaporwave, ...); see `art_critic/conditions.py`. No labels exist for these, so the 40 best CLIP matches stand in as the items that fit.
- **Labels.** If the media folder has a `labels.json` mapping each file to `{axis: value}`, every value with at least 12 items becomes a condition, and which items fit it is known exactly. WikiArt ships with `style`, which is used, and `artist` and `genre`, which are too specific and are skipped.
- **Your own.** Type anything into the box on the page, or pass `--conditions file.json` with `{"axis": ["value", ...]}` to add whole axes, for example modifiers pulled from your own prompts. These are matched by CLIP as well, and always show their 40 closest items, close or not.

Feedback given under a condition that is later dropped still trains the model; the condition just stops being offered.

By default the page rotates: three rounds on a condition, then on to the least-visited axis and value. The condition's axis and value under the images are menus: pick from them, or type your own, to pin one. `0` resumes rotation.

## Feedback

Any mix of these, all optional:

- mark an item your favorite of the four
- thumbs up or down on any item
- 1-5 stars on any item

A round with no feedback is logged as a skip and not trained on. Stars give the level; the favorite breaks ties when everything is a 5.

The condition can be switched off for the four on screen with the "no condition" button or `-`. It is struck out, and that round's feedback is logged as plain taste: it teaches the model what you like in general, under no prompt. The next round has its condition back.

Every control has its own key, so a round needs no mouse. The left hand covers the two left items and the right hand the two right:

| Item | Favorite | Stars 1-5 | Thumbs down / up |
|---|---|---|---|
| 1 (top left) | `1` | `Q W E R T` | `Z` / `X` |
| 2 (top right) | `2` | `Y U I O P` | `N` / `M` |
| 3 (bottom left) | `3` | `A S D F G` | `C` / `V` |
| 4 (bottom right) | `4` | `H J K L ;` | `,` / `.` |

Pressing a key again undoes it. `Enter` continues, `-` switches the condition off for this round, `/` jumps to the condition box, `Esc` leaves it, and `0` resumes auto-rotation. `?` prints each key under its control, or hides them again; they are hidden to start with. `\` opens the side panel with the chart and the model's predictions, which is also closed to start with so the images have the page to themselves.

## How it works

| Piece | File | What it does |
|---|---|---|
| Source | `art_critic/source.py` | A folder of images and videos, with optional labels, behind a small protocol |
| Embeddings | `art_critic/embed.py` | Frozen CLIP image and text embeddings; PCA for the reward model |
| Conditions | `art_critic/conditions.py` | The condition vocabulary |
| Vector store | `art_critic/index.py` | ChromaDB holds item and condition embeddings and answers retrieval |
| Reward model | `art_critic/reward.py` | Preference for an item given a condition, fit jointly on all three feedback types |
| Sampler | `art_critic/sampler.py` | Picks the next four: two top-scored, one most uncertain, one random |
| Feedback log | `art_critic/store.py` | Append-only `feedback.jsonl`: condition, items shown, feedback, both guesses |
| Loop | `art_critic/harness.py` | Ties the above together; `art_critic/server.py` exposes it over HTTP |

**The model.** The score for an item under a condition is

```
r(condition, item) = match + theta . [ match,  z(item),  z(condition) x z(item) ]
```

- `match` is CLIP similarity between the item and the condition's prompt. It is not learned, and it is all the model has before any feedback.
- `z(item)` carries general taste: what you like under any condition.
- `z(condition) x z(item)` carries how your taste shifts with the condition. Conditions with similar prompts share what is learned, so it reaches conditions you have never been asked about.

A favorite is a softmax choice over the four shown, a thumb is a Bernoulli, and a star rating is an ordinal logistic with learned cutpoints. All three are losses on the same score and are summed.

**Uncertainty.** `theta` has a Gaussian prior and the fit is the MAP estimate. A Laplace approximation gives every (condition, item) pair a score uncertainty, and the sampler spends one tile per round on the most uncertain item.

**Where ChromaDB comes in.** It stores the embeddings and answers three queries: which items carry a label (metadata filter), which items best match a text condition (inner-product search), and which conditions sit closest to your general taste (the same search over the conditions collection). Scoring and choosing the next round happen in memory.

**Video.** A video is embedded as the mean of six evenly spaced frames and played muted and looping in its tile. That captures how it looks, not how it moves.

## Tests

```sh
uv run pytest
```

## Notes

- Reset archives `feedback.jsonl` under a timestamped name; it does not delete it.
- WikiArt is for non-commercial research use.
- WikiArt has no digital art or photography, so conditions like those have nothing good to match. A text condition always returns its 40 closest items, whether or not they are close.
