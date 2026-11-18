"""HTTP surface over the harness, plus the static UI."""

from __future__ import annotations

from pathlib import Path

from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

from .harness import Harness

STATIC = Path(__file__).parent / "static"


class Page(StaticFiles):
    """The UI files, always revalidated, so a refresh never runs a stale script."""

    async def get_response(self, path, scope):
        response = await super().get_response(path, scope)
        response.headers["Cache-Control"] = "no-cache"
        return response


class Feedback(BaseModel):
    round_id: str
    pick: str | None = None
    thumbs: dict[str, int] = Field(default_factory=dict)
    ratings: dict[str, int] = Field(default_factory=dict)
    free: bool = False  # given with the condition switched off


def create_app(harness: Harness) -> FastAPI:
    app = FastAPI(title="art-critic")

    @app.get("/api/conditions")
    def get_conditions():
        return harness.vocabulary()

    @app.get("/api/round")
    def get_round(axis: str | None = None, value: str | None = None, text: str | None = None):
        """No arguments: the harness rotates conditions. axis+value pins one; text makes a custom one."""
        try:
            return harness.new_round(axis, value, text)
        except KeyError:
            raise HTTPException(404, f"no such condition: {axis}: {value}")

    @app.post("/api/feedback")
    def post_feedback(body: Feedback):
        try:
            return harness.submit(body.round_id, body.pick, body.thumbs, body.ratings, body.free)
        except KeyError as err:
            raise HTTPException(409, str(err.args[0]))
        except ValueError as err:
            raise HTTPException(422, str(err))

    @app.post("/api/reset")
    def post_reset():
        harness.reset()
        return {"ok": True}

    def serve(item_id: str, which: str):
        try:
            path = getattr(harness.source, which)(item_id)
        except KeyError:
            raise HTTPException(404, "no such item")
        return FileResponse(path, headers={"Cache-Control": "max-age=86400"})

    @app.get("/media/{item_id}")
    def get_media(item_id: str):
        return serve(item_id, "path")

    @app.get("/poster/{item_id}")
    def get_poster(item_id: str):
        return serve(item_id, "poster")

    app.mount("/", Page(directory=STATIC, html=True), name="static")
    return app
