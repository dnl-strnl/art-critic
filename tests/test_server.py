"""End to end through HTTP, Chroma and the feedback log, with fake embedders."""

import hashlib
import json

import av
import numpy as np
import pytest
from fastapi.testclient import TestClient
from PIL import Image

from art_critic.__main__ import build_harness
from art_critic.harness import STICK, TEXT_MEMBERS
from art_critic.server import create_app

PER_HUE = 24
POOL = 2 * PER_HUE + 1  # stills plus one video
WARM = {"axis": "hue", "value": "warm"}


def fake_image_embed(paths):
    """Embedding = a fixed random lift of the image's mean color."""
    lift = np.random.default_rng(0).standard_normal((3, 32))
    out = []
    for p in paths:
        with Image.open(p) as im:
            rgb = np.asarray(im.convert("RGB"), dtype=np.float32).mean(axis=(0, 1)) / 255
        v = (rgb - 0.5) @ lift
        out.append(v / np.linalg.norm(v))
    return np.array(out, dtype=np.float32)


def fake_text_embed(texts):
    out = []
    for t in texts:
        seed = int(hashlib.sha1(t.encode()).hexdigest()[:8], 16)
        v = np.random.default_rng(seed).standard_normal(32)
        out.append(v / np.linalg.norm(v))
    return np.array(out, dtype=np.float32)


def make_video(path, frames=24):
    with av.open(str(path), "w") as container:
        stream = container.add_stream("mpeg4", rate=12)
        stream.width, stream.height, stream.pix_fmt = 64, 64, "yuv420p"
        for i in range(frames):
            pixels = np.full((64, 64, 3), (250 - i * 4, 60, 30 + i * 4), dtype=np.uint8)
            for packet in stream.encode(av.VideoFrame.from_ndarray(pixels, format="rgb24")):
                container.mux(packet)
        for packet in stream.encode():
            container.mux(packet)


@pytest.fixture
def dirs(tmp_path):
    images = tmp_path / "images"
    images.mkdir()
    rng = np.random.default_rng(1)
    labels = {}
    for i in range(2 * PER_HUE):
        warm = i % 2 == 0
        strong, weak = int(rng.integers(150, 256)), int(rng.integers(0, 100))
        color = (strong, int(rng.integers(0, 256)), weak) if warm else (weak, int(rng.integers(0, 256)), strong)
        Image.new("RGB", (32, 32), color).save(images / f"{i:02d}.png")
        labels[f"{i:02d}.png"] = {"hue": "warm" if warm else "cool"}
    make_video(images / "clip.mp4")
    labels["clip.mp4"] = {"hue": "warm"}
    (images / "labels.json").write_text(json.dumps(labels))
    return images, tmp_path / "data"


def make_client(dirs):
    images, data = dirs
    harness = build_harness(images, data, embed_images=fake_image_embed, embed_texts=fake_text_embed, seed=0)
    return TestClient(create_app(harness)), harness


def redness(harness, item_id):
    with Image.open(harness.source.poster(item_id)) as im:
        r, g, b = np.asarray(im.convert("RGB"), dtype=np.float32).mean(axis=(0, 1))
    return r - (g + b) / 2


def play(client, harness, rounds, free=False, **condition):
    """A person who likes red: picks the reddest, thumbs and stars by redness."""
    for _ in range(rounds):
        rnd = client.get("/api/round", params=condition).json()
        ids = [it["id"] for it in rnd["items"]]
        red = {i: redness(harness, i) for i in ids}
        body = {
            "round_id": rnd["round_id"],
            "pick": max(ids, key=red.get),
            "thumbs": {i: 1 if red[i] > 0 else -1 for i in ids[:2]},
            "ratings": {i: int(np.clip(3 + red[i] / 50, 1, 5)) for i in ids[2:]},
            "free": free,
        }
        res = client.post("/api/feedback", json=body)
        assert res.status_code == 200, res.text
    return client.get("/api/round", params=condition).json()["state"]


def test_vocabulary_has_label_and_text_axes(dirs):
    client, _ = make_client(dirs)
    vocab = client.get("/api/conditions").json()
    assert vocab["hue"] == ["cool", "warm"]
    assert "serene" in vocab["vibe"] and "minimal" in vocab["aesthetic"]


def test_specific_label_axes_are_not_conditions(dirs):
    images = dirs[0]
    labels = json.loads((images / "labels.json").read_text())
    (images / "labels.json").write_text(json.dumps({name: {**label, "artist": "Someone"} for name, label in labels.items()}))
    client, _ = make_client(dirs)
    assert set(client.get("/api/conditions").json()) == {"hue", "vibe", "aesthetic"}
    assert client.get("/api/round", params={"axis": "artist", "value": "Someone"}).status_code == 404


def test_labeled_condition_shows_only_items_with_the_label(dirs):
    client, harness = make_client(dirs)
    assert client.get("/").status_code == 200
    rnd = client.get("/api/round", params=WARM).json()
    assert rnd["condition"] == {"key": "hue:warm", "axis": "hue", "value": "warm", "prompt": "warm", "auto": False, "labeled": True}
    assert rnd["state"]["members"] == PER_HUE + 1 and not rnd["state"]["fitted"]
    for _ in range(3):
        rnd = client.get("/api/round", params=WARM).json()
        assert all(harness.source.labels(it["id"])["hue"] == "warm" for it in rnd["items"])


def test_text_condition_uses_its_closest_matches(dirs):
    client, harness = make_client(dirs)
    rnd = client.get("/api/round", params={"axis": "vibe", "value": "serene"}).json()
    assert not rnd["condition"]["labeled"] and rnd["state"]["members"] == TEXT_MEMBERS
    members = {harness.ids[r] for r in harness.members("vibe:serene")}
    match = harness.match("vibe:serene")
    worst_member = min(match[harness.row[i]] for i in members)
    assert all(match[harness.row[i]] <= worst_member for i in harness.ids if i not in members)
    assert {it["id"] for it in rnd["items"]} <= members


def test_feedback_is_logged_with_its_condition_and_trains_the_model(dirs):
    client, harness = make_client(dirs)
    state = play(client, harness, 5, **WARM)
    assert state["fitted"]
    assert state["counts"] == {"rounds": 5, "skipped": 0, "picks": 5, "thumbs": 10, "ratings": 10, "conditions": 1}
    assert state["unseen"] == PER_HUE + 1 - 20
    assert len(state["accuracy"]["model"]) == len(state["accuracy"]["match"]) == 5
    assert [row["value"] for row in state["by_condition"]["hue"]] == ["warm"]
    assert len(state["lean"]["likes"]) == 6

    # Under the condition, unseen favorites are redder than unseen least-liked.
    fav = np.mean([redness(harness, it["id"]) for it in state["favorites"][:2]])
    least = np.mean([redness(harness, it["id"]) for it in state["least"][:2]])
    assert fav > least

    log = [json.loads(line) for line in (dirs[1] / "feedback.jsonl").read_text().splitlines()]
    assert len(log) == 5
    for event in log:
        assert event["condition"] == {"key": "hue:warm", "axis": "hue", "value": "warm", "prompt": "warm"}
        assert len(event["shown"]) == 4 and event["model_pick"] in event["shown"] and event["match_pick"] in event["shown"]


def test_feedback_under_one_condition_carries_to_another(dirs):
    """General taste: liking red under 'warm' should rank red first under 'cool' too."""
    client, harness = make_client(dirs)
    play(client, harness, 6, **WARM)
    cool = harness.members("hue:cool")
    mean, _ = harness.score("hue:cool", cool)
    red = np.array([redness(harness, harness.ids[r]) for r in cool])
    assert np.corrcoef(mean, red)[0, 1] > 0.5


def test_auto_rotation_sticks_then_moves_to_a_new_axis(dirs):
    client, _ = make_client(dirs)
    seen = []
    for _ in range(STICK + 1):
        rnd = client.get("/api/round").json()
        assert rnd["condition"]["auto"]
        seen.append(rnd["condition"])
        client.post("/api/feedback", json={"round_id": rnd["round_id"]})
    assert len({c["key"] for c in seen[:STICK]}) == 1
    assert seen[STICK]["axis"] != seen[0]["axis"]


def test_custom_condition_is_created_and_survives_restart(dirs):
    client, harness = make_client(dirs)
    rnd = client.get("/api/round", params={"text": "  vaporwave nostalgia "}).json()
    assert rnd["condition"]["axis"] == "custom" and rnd["condition"]["value"] == "vaporwave nostalgia"
    assert rnd["state"]["members"] == TEXT_MEMBERS
    before = play(client, harness, 3, text="vaporwave nostalgia")
    del client, harness

    client, harness = make_client(dirs)
    assert client.get("/api/conditions").json()["custom"] == ["vaporwave nostalgia"]
    assert len(harness.events) == 3
    assert client.get("/api/round", params={"text": "vaporwave nostalgia"}).json()["state"] == before


def test_video_is_one_item_embedded_from_its_frames(dirs):
    client, harness = make_client(dirs)
    (video,) = [i for i in harness.ids if harness.source.kind(i) == "video"]
    assert len(harness.source.frames(video)) == 6
    # Frames run red to blue; the item's vector should sit between its first and last frame.
    first, last = fake_image_embed([harness.source.frames(video)[0], harness.source.frames(video)[-1]])
    vector = harness.X[harness.row[video]]
    assert vector @ first > first @ last and vector @ last > first @ last

    media = client.get(f"/media/{video}")
    assert media.status_code == 200 and media.headers["content-type"] == "video/mp4"
    poster = client.get(f"/poster/{video}")
    assert poster.status_code == 200 and poster.headers["content-type"] == "image/jpeg"

    for _ in range(8):  # it turns up in rounds like any other item
        rnd = client.get("/api/round", params=WARM).json()
        hit = [it for it in rnd["items"] if it["id"] == video]
        client.post("/api/feedback", json={"round_id": rnd["round_id"], "pick": rnd["items"][0]["id"]})
        if hit:
            assert hit[0]["kind"] == "video" and hit[0]["url"] == f"/media/{video}"
            break
    else:
        pytest.fail("video never shown")


def test_skip_is_logged_but_not_trained_on(dirs):
    client, _ = make_client(dirs)
    rnd = client.get("/api/round", params=WARM).json()
    assert client.post("/api/feedback", json={"round_id": rnd["round_id"]}).json() == {"model_hit": None, "match_hit": None}
    state = client.get("/api/round", params=WARM).json()["state"]
    assert state["counts"]["skipped"] == 1 and not state["fitted"] and state["unseen"] == PER_HUE + 1 - 4


def test_rejects_bad_requests(dirs):
    client, harness = make_client(dirs)
    rnd = client.get("/api/round", params=WARM).json()
    assert client.post("/api/feedback", json={"round_id": "nope"}).status_code == 409
    stranger = next(i for i in harness.ids if i not in {it["id"] for it in rnd["items"]})
    assert client.post("/api/feedback", json={"round_id": rnd["round_id"], "pick": stranger}).status_code == 422
    assert client.get("/api/round", params={"axis": "hue", "value": "plaid"}).status_code == 404
    assert client.get("/media/nope").status_code == 404
    assert not (dirs[1] / "feedback.jsonl").exists()


def test_reset_archives_and_clears(dirs):
    client, harness = make_client(dirs)
    play(client, harness, 3, **WARM)
    assert client.post("/api/reset").status_code == 200
    state = client.get("/api/round", params=WARM).json()["state"]
    assert not state["fitted"] and state["unseen"] == PER_HUE + 1 and state["counts"]["rounds"] == 0
    archived = list(dirs[1].glob("feedback-*.jsonl"))
    assert len(archived) == 1 and len(archived[0].read_text().splitlines()) == 3


def test_condition_wraps_when_exhausted(dirs):
    client, harness = make_client(dirs)
    state = play(client, harness, (PER_HUE + 1) // 4 + 2, **WARM)
    assert 4 <= state["unseen"] <= PER_HUE + 1


def test_condition_can_be_switched_off_for_a_round(dirs):
    """The same four items, judged as plain taste: logged under no condition, and still learned from."""
    client, harness = make_client(dirs)
    cool = harness.members("hue:cool")
    red = np.array([redness(harness, harness.ids[r]) for r in cool])
    untrained = np.corrcoef(harness.score("hue:cool", cool)[0], red)[0, 1]
    state = play(client, harness, 5, free=True, **WARM)

    log = [json.loads(line) for line in (dirs[1] / "feedback.jsonl").read_text().splitlines()]
    assert all(e["condition"]["key"] == "open:no condition" and e["dealt_under"]["key"] == "hue:warm" for e in log)
    assert all(e["model_pick"] in e["shown"] and e["match_pick"] in e["shown"] and "open" not in e for e in log)
    # It is not a condition anyone can pick, and it does not count as one.
    assert "open" not in client.get("/api/conditions").json()
    assert client.get("/api/round", params={"axis": "open", "value": "no condition"}).status_code == 404
    assert state["counts"]["rounds"] == 5 and state["counts"]["conditions"] == 0
    # The items still count as seen under the condition they were dealt for.
    assert state["unseen"] == PER_HUE + 1 - 20
    # General taste was learned: red moves up the ranking under a condition never judged.
    assert state["fitted"] and np.corrcoef(harness.score("hue:cool", cool)[0], red)[0, 1] > untrained + 0.5

    del client, harness
    client, harness = make_client(dirs)
    assert len(harness.events) == 5 and harness.model.fitted


def test_feedback_under_a_retired_condition_still_trains(dirs):
    """A condition that is dropped from the vocabulary keeps its rounds, but is no longer offered."""
    client, harness = make_client(dirs)
    cool = harness.members("hue:cool")
    red = np.array([redness(harness, harness.ids[r]) for r in cool])
    untrained = np.corrcoef(harness.score("hue:cool", cool)[0], red)[0, 1]
    play(client, harness, 5, **WARM)
    log = dirs[1] / "feedback.jsonl"
    gone = {"key": "era:medieval", "axis": "era", "value": "medieval", "prompt": "a medieval artwork"}
    log.write_text("".join(json.dumps({**json.loads(line), "condition": gone}) + "\n" for line in log.read_text().splitlines()))
    del client, harness

    client, harness = make_client(dirs)
    assert len(harness.events) == 5 and harness.model.fitted
    assert "era" not in client.get("/api/conditions").json()
    assert client.get("/api/round", params={"axis": "era", "value": "medieval"}).status_code == 404
    for _ in range(2 * STICK):
        rnd = client.get("/api/round").json()
        assert rnd["condition"]["axis"] != "era"
        client.post("/api/feedback", json={"round_id": rnd["round_id"]})
    # What was learned under it still pulls red up the ranking elsewhere.
    assert np.corrcoef(harness.score("hue:cool", cool)[0], red)[0, 1] > untrained + 0.25


def test_rounds_logged_without_a_condition_are_left_alone(dirs):
    data = dirs[1]
    data.mkdir()
    legacy = json.dumps({"round_id": "old", "shown": ["a", "b"], "pick": "a", "thumbs": {}, "ratings": {}, "skipped": False})
    (data / "feedback.jsonl").write_text(legacy + "\n")
    client, harness = make_client(dirs)
    assert harness.events == []
    play(client, harness, 1, **WARM)
    lines = (data / "feedback.jsonl").read_text().splitlines()
    assert lines[0] == legacy and len(lines) == 2
