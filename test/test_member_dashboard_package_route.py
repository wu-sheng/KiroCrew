"""End to end: a fold moves, and the field the page subscribed to arrives on the socket.

Every other test of this feature holds one piece still. This one drives the whole chain
the way the product does -- the real route, the real artifact store, the real crew-log
bus, the real owner gate, the real broadcast -- because the pieces were each correct
while nothing connected them, which is exactly what a unit test cannot see.

Four claims:

1. The route serves a ``kind="dashboard"`` package's own page, composed through
   ``dashboard_frame``, and arms the live page as it does so.
2. A fold advancing puts the SUBSCRIBED field on the owner's socket, and a key the
   model does not declare is not on it. The model is the allowlist, and this is the
   only test where "not pushed" is measured on the wire rather than in a dict.
3. A page that cannot place a patch asks for a whole read, and the route answers it.
4. A worker is refused at the real artifact write path, with the sentence that tells
   it what to do instead.
"""

from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace
from typing import Any
from unittest.mock import patch

import pytest
from aiohttp import web
from aiohttp.test_utils import make_mocked_request
from chat_test_helpers import _make_state

from kiro_crew.artifact_store import dashboard_package as dp
from kiro_crew.config.loader import KiroCrewAgentConfig
from kiro_crew.crew_log import bus as crew_log_bus
from kiro_crew.dashboard.handlers import artifacts as artifacts_handler
from kiro_crew.dashboard.handlers import member_dashboard as md
from kiro_crew.dashboard.handlers import member_dashboard_fields as mdf

CREW = "atlas"
SLUG = "atlas"
OWNER = "U_OWNER"
PACKAGE_SLUG = "atlas-dashboard"

#: Above any revision the eager folder has already minted for the slot: the feed
#: subscribes with a baseline, and the cache drops an event that is not newer.
_REVISION_BASE = 2_000_000

PAGE = (
    '<p data-dashboard-field="goal"></p>'
    "<script>window.kirocrew.subscribe('goal', (v) => v);</script>"
)


def _package(bound_to: str = f"crewmate:{SLUG}") -> dict[str, Any]:
    return dp.validate_package(
        {
            "kind": "dashboard",
            "bound_to": bound_to,
            "model": {
                "types": {
                    "goal": {
                        "type": "text",
                        "label": "Goal",
                        "source": {"fold": "work", "path": "conductor.goal"},
                    }
                }
            },
            "view": {"html": PAGE},
            "theme": {"tokens": {}},
        }
    )


def _fake_cfg() -> Any:
    return SimpleNamespace(
        agents={CREW: KiroCrewAgentConfig(kiro_agent="kirocrew", description="d", model="m1")},
        default_agent=CREW,
        memory_stores={},
        workspaces={"default": SimpleNamespace(dir="workspace")},
        default_workspace="default",
        degraded_sections=frozenset(),
    )


class _Store:
    """The artifact store, narrowed to what the route reads: one package by slug."""

    def __init__(self, content: str) -> None:
        self.content = content

    def get(self, slug: str) -> Any:
        if slug != PACKAGE_SLUG:
            from kiro_crew.artifacts import ArtifactError

            raise ArtifactError(f"no artifact {slug!r}")
        return SimpleNamespace(slug=slug, content=self.content, version=4, kind="dashboard")


@pytest.fixture()
def wired(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Any:
    """The route's app, with the frames it broadcasts recorded."""
    frames: list[tuple[str, Any]] = []
    state = _make_state(tmp_path)
    state.owner_id = OWNER
    monkeypatch.setattr(
        state, "broadcast_ws_owners", lambda kind, data: frames.append((kind, data)), raising=False
    )
    app = web.Application()
    app["state"] = state
    store = _Store(dp.dump_package(_package()))
    monkeypatch.setattr("kiro_crew.artifacts.get_default_store", lambda: store)
    # The slot the fold values are keyed by, resolved rather than guessed in
    # production; pinned here so the publish below names the same one.
    monkeypatch.setattr(md, "_dashboard_slot", lambda member, slug: SLUG)
    monkeypatch.setattr(md, "_write_session", lambda slug, member: "")
    monkeypatch.setattr("kiro_crew.members.member_slug", lambda name, config=None: SLUG)
    yield SimpleNamespace(app=app, frames=frames, store=store)
    for page in list(app.get("member_dashboard_live_pages", {}).values()):
        page.disarm()


def _request(app: web.Application, method: str, path: str) -> web.Request:
    req = make_mocked_request(method, path, app=app, match_info={"slug": SLUG})
    req["app"] = ""
    req["user"] = OWNER
    return req


async def _call(handler: Any, req: web.Request) -> tuple[int, Any]:
    with patch.object(md.KiroCrewConfig, "load", return_value=_fake_cfg()):
        resp = await handler(req)
    return resp.status, json.loads(resp.body)


def _publish(fold: str, value: dict[str, Any], revision: int) -> None:
    crew_log_bus.publish(
        crew_log_bus.FOLD_ADVANCED,
        crew_log_bus.FoldAdvanced(
            scope=crew_log_bus.SCOPE_SLOT,
            key=SLUG,
            fold=fold,
            revision=_REVISION_BASE + revision,
            seq=revision,
            value=value,
        ),
    )


PATH = f"/api/members/{SLUG}/dashboard?member={CREW}&dashboard={PACKAGE_SLUG}"


@pytest.mark.asyncio
async def test_the_route_serves_the_owners_own_page_through_the_frame(wired: Any) -> None:
    """The page is composed, not handed over raw: a data island, the bootstrap that
    freezes it onto ``window``, and the author's markup verbatim inside the host's
    body. Served under the key the tab already mounts."""
    status, body = await _call(md.api_member_dashboard, _request(wired.app, "GET", PATH))
    assert status == 200
    page = body["rendered_html"]
    assert PAGE in page, "the page is not placed verbatim"
    assert "kirocrew-dashboard-data" in page
    assert body["fields"] == ["goal"], "the body does not name the allowlist"
    assert body["instance_version"] == 4


@pytest.mark.asyncio
async def test_the_route_arms_the_live_page(wired: Any) -> None:
    await _call(md.api_member_dashboard, _request(wired.app, "GET", PATH))
    pages = wired.app["member_dashboard_live_pages"]
    assert SLUG in pages
    assert pages[SLUG].folds() == frozenset({"work"})


@pytest.mark.asyncio
async def test_a_fold_puts_the_subscribed_field_on_the_owners_socket(wired: Any) -> None:
    """THE WHOLE CHAIN. Nothing in this test calls the controller: the route arms it,
    the bus carries a real fold event, and the frame is read off the broadcast."""
    await _call(md.api_member_dashboard, _request(wired.app, "GET", PATH))
    wired.frames.clear()
    _publish("work", {"conductor": {"goal": "ship the host"}, "omitted": 0}, 1)
    assert wired.frames, "a fold moved and nothing reached the socket"
    kind, data = wired.frames[-1]
    assert kind == mdf.FIELD_FRAME
    assert data["slug"] == SLUG
    assert data["read"]["fields"]["goal"] == "ship the host"
    assert data["read"]["patch"] is True


@pytest.mark.asyncio
async def test_a_key_the_model_does_not_declare_is_not_on_the_wire(wired: Any) -> None:
    """Measured on the BROADCAST, which is the only place it counts. A work ledger's
    fold value carries every item and every worker's report; a page binding one field
    of it must not be handed the rest, and the model is what says so."""
    await _call(md.api_member_dashboard, _request(wired.app, "GET", PATH))
    wired.frames.clear()
    _publish(
        "work",
        {"conductor": {"goal": "ship the host"}, "items": ["a secret item"], "omitted": 3},
        1,
    )
    sent = json.dumps(wired.frames[-1][1])
    assert "a secret item" not in sent
    assert set(wired.frames[-1][1]["read"]["fields"]) == {"goal"}


@pytest.mark.asyncio
async def test_a_fold_that_moves_without_the_field_changing_sends_nothing(wired: Any) -> None:
    await _call(md.api_member_dashboard, _request(wired.app, "GET", PATH))
    _publish("work", {"conductor": {"goal": "ship the host"}}, 1)
    wired.frames.clear()
    _publish("work", {"conductor": {"goal": "ship the host"}, "items": ["new"]}, 2)
    assert wired.frames == []


@pytest.mark.asyncio
async def test_each_frame_steps_the_version_by_one(wired: Any) -> None:
    """What the page checks a patch against. A counter that skipped would make every
    page ask for a full read on every push."""
    await _call(md.api_member_dashboard, _request(wired.app, "GET", PATH))
    wired.frames.clear()
    for n, goal in enumerate(("one", "two", "three"), start=1):
        _publish("work", {"conductor": {"goal": goal}}, n)
    versions = [data["read"]["version"] for _kind, data in wired.frames]
    assert versions == list(range(versions[0], versions[0] + len(versions)))


@pytest.mark.asyncio
async def test_the_refetch_route_answers_with_the_whole_read(wired: Any) -> None:
    await _call(md.api_member_dashboard, _request(wired.app, "GET", PATH))
    _publish("work", {"conductor": {"goal": "ship the host"}}, 1)
    status, body = await _call(
        md.api_member_dashboard_refetch,
        _request(wired.app, "POST", f"/api/members/{SLUG}/dashboard/refetch?member={CREW}"),
    )
    assert status == 200
    assert body["read"]["fields"]["goal"] == "ship the host"
    assert "patch" not in body["read"], "a full read must replace, not merge"


@pytest.mark.asyncio
async def test_the_refetch_route_says_so_when_nothing_is_armed(wired: Any) -> None:
    status, body = await _call(
        md.api_member_dashboard_refetch,
        _request(wired.app, "POST", f"/api/members/{SLUG}/dashboard/refetch?member={CREW}"),
    )
    assert status == 409
    assert body["code"] == "no_live_page"


@pytest.mark.asyncio
async def test_a_package_bound_elsewhere_is_not_served_here(wired: Any) -> None:
    """A page carries the binding's own fold values, so rendering one bound elsewhere
    would show one crewmate's work under another's name."""
    wired.store.content = dp.dump_package(_package(bound_to="crewmate:nyx"))
    status, body = await _call(md.api_member_dashboard, _request(wired.app, "GET", PATH))
    assert status == 403
    assert body["code"] == "dashboard_not_bound"


@pytest.mark.asyncio
async def test_a_non_owner_is_refused_before_the_package_is_read(wired: Any) -> None:
    req = _request(wired.app, "GET", PATH)
    req["user"] = "123456789"

    def _no_read() -> Any:
        raise AssertionError("a refused caller must not reach the artifact store")

    with patch("kiro_crew.artifacts.get_default_store", side_effect=_no_read):
        status, body = await _call(md.api_member_dashboard, req)
    assert status == 403
    assert body["code"] == "owner_only"


class TestTheWorkerIsRefusedAtTheRealWritePath:
    """The owner gate on the path a worker's ``artifact_update`` actually takes."""

    def _request(self, app: web.Application, crew: str) -> web.Request:
        req = make_mocked_request(
            "PATCH",
            f"/api/artifacts/{PACKAGE_SLUG}",
            headers={"X-Internal-Secret": "s", "X-Session-Key": "chat-9-17911"},
            app=app,
            match_info={"slug": PACKAGE_SLUG},
        )
        app["state"].sessions = SimpleNamespace(get_agent_selection=lambda key: ("member", crew))
        return req

    def test_a_worker_is_refused_with_the_sentence_that_says_what_to_do(
        self, wired: Any, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setattr("kiro_crew.members.member_slug", lambda name, config=None: name.lower())
        monkeypatch.setattr(mdf, "writing_caller", lambda request: ("nyx", "chat-9-17911"))
        said = mdf.package_write_refusal(
            self._request(wired.app, "nyx"), dp.dump_package(_package())
        )
        assert said, "a worker may write another crewmate's dashboard package"
        assert "atlas" in said
        assert "work_report" in said

    def test_the_owner_is_not_refused(self, wired: Any, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setattr("kiro_crew.members.member_slug", lambda name, config=None: name.lower())
        monkeypatch.setattr(mdf, "writing_caller", lambda request: ("atlas", "chat-1-1790"))
        assert (
            mdf.package_write_refusal(
                self._request(wired.app, "atlas"), dp.dump_package(_package())
            )
            == ""
        )

    def test_a_document_that_is_not_a_package_is_not_gated(self, wired: Any) -> None:
        """Asked on every artifact write, so refusing a prose document for having no
        binding would break every other kind."""
        req = self._request(wired.app, "nyx")
        assert mdf.package_write_refusal(req, "# just a note") == ""
        assert mdf.package_write_refusal(req, None) == ""

    def test_the_update_route_calls_the_gate(self) -> None:
        """The gate is only a gate if the route asks it. Pinned on the SOURCE of the
        three write paths, because a route that stopped asking would leave every test
        above passing."""
        source = Path(artifacts_handler.__file__).read_text(encoding="utf-8")
        assert source.count("package_write_refusal(") == 3, (
            "expected the gate on all three write paths -- api_artifact_update, the "
            "create that lands on an existing slug, and the fresh create. Whichever "
            "one goes unasked is the spelling a worker would use"
        )
        assert "from kiro_crew.dashboard.handlers.member_dashboard_fields import" in source


# --------------------------------------------------------------------------- #
# The tree fold, through the production route. This is what makes
# `read_tree_projection` reachable at all: it has no bus scope, so nothing can push
# it, and the controller reads it on arm and on refetch.
# --------------------------------------------------------------------------- #

GALLERY = (
    Path(mdf.__file__).resolve().parents[2]
    / "builtin_skills"
    / "dashboard"
    / "gallery"
    / "dispatch-tree"
    / "package.json"
)


def _tree_value(goal: str, depth: int = 0) -> dict[str, Any]:
    """One rendered ``worktree`` projection, in the fold's own declared shape."""
    return {
        "boards": [{"slot_key": "chat-1-1790", "goal": goal, "depth": depth, "entries": 2}],
        "roots": ["chat-1-1790"],
        "cycles": 0,
        "unresolved": 0,
        "dropped": 0,
        "goal_chars": len(goal),
        "limit": 64,
        "owner_limit": 8,
        "root": SLUG,
    }


@pytest.fixture()
def tree_wired(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Any:
    """The route, serving the GALLERY package, with the tree read intercepted.

    The gallery page is loaded off disk rather than rebuilt here: it is the artifact a
    reader is told to copy, so a test that served a hand-written stand-in would prove
    the route works for a package nobody has.

    ``read_tree_projection`` itself is stubbed because its real walk needs a fleet of
    on-disk logs; what this pins is that the ROUTE calls it, with the root it should,
    and puts its value on the page. :mod:`test_crew_log_worktree_read` owns the walk.
    """
    frames: list[tuple[str, Any]] = []
    calls: list[tuple[str, str]] = []
    value = {"v": _tree_value("ship the host")}
    state = _make_state(tmp_path)
    state.owner_id = OWNER
    monkeypatch.setattr(
        state, "broadcast_ws_owners", lambda kind, data: frames.append((kind, data)), raising=False
    )
    app = web.Application()
    app["state"] = state
    store = _Store(GALLERY.read_text(encoding="utf-8"))
    monkeypatch.setattr("kiro_crew.artifacts.get_default_store", lambda: store)
    monkeypatch.setattr(md, "_dashboard_slot", lambda member, slug: SLUG)
    monkeypatch.setattr(md, "_write_session", lambda slug, member: "")
    monkeypatch.setattr("kiro_crew.members.member_slug", lambda name, config=None: SLUG)
    # The gallery package ships bound to the example crewmate; this route serves the
    # crewmate it is bound to, so the binding is the thing a copier changes first.
    store.content = store.content.replace('"crewmate:example"', f'"crewmate:{SLUG}"')

    def _read_tree(root: str, name: str) -> Any:
        calls.append((root, name))
        return SimpleNamespace(value=value["v"], seq=1)

    monkeypatch.setattr("kiro_crew.crew_log.projection.read_tree_projection", _read_tree)
    yield SimpleNamespace(app=app, frames=frames, calls=calls, value=value, store=store)
    for page in list(app.get("member_dashboard_live_pages", {}).values()):
        page.disarm()


TREE_PATH = f"/api/members/{SLUG}/dashboard?member={CREW}&dashboard={PACKAGE_SLUG}"


@pytest.mark.asyncio
async def test_the_route_reads_the_tree_fold_for_this_slot(tree_wired: Any) -> None:
    """THE PRODUCTION CALLER of ``read_tree_projection``. Asked with the page's own slot
    as the root, because a tree walks what the record says from one root outward -- the
    boards that root's units bound, and the boards those conduct, to closure."""
    status, _body = await _call(md.api_member_dashboard, _request(tree_wired.app, "GET", TREE_PATH))
    assert status == 200
    assert (SLUG, "worktree") in tree_wired.calls


@pytest.mark.asyncio
async def test_the_tree_value_is_on_the_page_the_route_serves(tree_wired: Any) -> None:
    status, body = await _call(md.api_member_dashboard, _request(tree_wired.app, "GET", TREE_PATH))
    assert status == 200
    island = body["rendered_html"].split('id="kirocrew-dashboard-data">')[1].split("</script>")[0]
    read = json.loads(island.replace("\\u003c", "<").replace("\\u003e", ">"))
    assert read["fields"]["boards"][0]["goal"] == "ship the host"
    assert read["fields"]["roots"] == ["chat-1-1790"]


@pytest.mark.asyncio
async def test_a_tree_bound_page_does_not_fly_the_stale_band(tree_wired: Any) -> None:
    """The feed reports a tree binding unavailable, because it has no scope to key it
    by. If the controller left those fields in ``missing`` the page would raise the band
    permanently -- over values that are current -- and a band a reader learns to ignore
    is worse than no band."""
    _status, body = await _call(md.api_member_dashboard, _request(tree_wired.app, "GET", TREE_PATH))
    island = body["rendered_html"].split('id="kirocrew-dashboard-data">')[1].split("</script>")[0]
    read = json.loads(island.replace("\\u003c", "<").replace("\\u003e", ">"))
    assert read["stale"] is False
    assert "boards" not in read["missing"]


@pytest.mark.asyncio
async def test_a_refetch_re_reads_the_tree(tree_wired: Any) -> None:
    """The only moment a tree value moves. Nothing can push one, so a refetch is not
    merely the gap remedy here -- it is the whole update path for this fold."""
    await _call(md.api_member_dashboard, _request(tree_wired.app, "GET", TREE_PATH))
    tree_wired.value["v"] = _tree_value("ship the gallery", depth=1)
    _status, body = await _call(
        md.api_member_dashboard_refetch,
        _request(tree_wired.app, "POST", f"/api/members/{SLUG}/dashboard/refetch?member={CREW}"),
    )
    assert body["read"]["fields"]["boards"][0]["goal"] == "ship the gallery"
    assert tree_wired.calls.count((SLUG, "worktree")) == 2


@pytest.mark.asyncio
async def test_a_work_ledger_push_still_rides_beside_the_tree(tree_wired: Any) -> None:
    """The gallery page binds BOTH: the tree for its shape and the work ledger for the
    counts. One fold having no event must not cost the other its push."""
    await _call(md.api_member_dashboard, _request(tree_wired.app, "GET", TREE_PATH))
    tree_wired.frames.clear()
    _publish("work", {"items": [{"id": "it_1"}, {"id": "it_2"}]}, 1)
    assert tree_wired.frames, "the work fold moved and nothing reached the socket"
    assert len(tree_wired.frames[-1][1]["read"]["fields"]["work_items"]) == 2


@pytest.mark.asyncio
async def test_the_gallery_pages_own_subscriber_receives_the_tree(tree_wired: Any) -> None:
    """END TO END, through the page's own script.

    The route reads the tree, composes the document, and the gallery page's own
    ``window.kirocrew.subscribe('boards', draw)`` is called with the boards that came
    out of ``read_tree_projection``. Everything between is the product's: the island,
    the bootstrap, the subscriber registry.

    The document's scripts are EXECUTED in Node under the same DOM shim
    ``test_dashboard_frame_subscribe`` uses, because the claim is about what runs. A
    page that composed correctly and never called its own callback renders an empty
    tree, and no assertion on the html would see it.
    """
    from test_dashboard_frame_subscribe import _run

    _status, body = await _call(md.api_member_dashboard, _request(tree_wired.app, "GET", TREE_PATH))
    island = body["rendered_html"].split('id="kirocrew-dashboard-data">')[1].split("</script>")[0]
    read = json.loads(island.replace("\\u003c", "<").replace("\\u003e", ">"))
    # The page's own script is replayed verbatim out of the stored package, so what is
    # exercised is the gallery artifact rather than a line retyped in this test.
    page_js = (
        dp.parse_package(tree_wired.store.content)["view"]["html"]
        .split("<script>")[1]
        .split("</script>")[0]
    )
    calls = _run(
        __import__("kiro_crew.dashboard_frame", fromlist=["x"])._BOOTSTRAP_JS,
        read,
        # The shim answers only for the data island, so the page's own two elements and
        # `createElement` are supplied here, RECORDING instead of building: what this
        # asserts is that the callback fired with the tree the route read, which is the
        # link the ruling is about. The row markup itself is pinned by the gallery test.
        "const node = () => ({"
        "  setAttribute() {}, appendChild(c) { if (c && c.text !== undefined)"
        "    __calls.push(c.text); }, style: {}, hidden: false,"
        "  set textContent(v) { this.text = v; }, get textContent() { return ''; },"
        "});"
        "const owned = { nodes: node(), empty: node() };"
        "const island = globalThis.document.getElementById;"
        "globalThis.document.getElementById = (id) => owned[id] || island(id);"
        "globalThis.document.createElement = () => node();" + page_js,
    )
    assert "ship the host" in calls, (
        "the gallery page's subscriber never received the tree the route read, so the "
        "page renders an empty tree over a fleet the record knows about"
    )
    assert "chat-1-1790" in calls


# --------------------------------------------------------------------------- #
# What leaves the gateway for a page. The wildcard `postMessage` target is
# unavoidable for an opaque-origin frame, so the two properties that make it
# safe are pinned here rather than argued in a comment.
# --------------------------------------------------------------------------- #


@pytest.mark.asyncio
async def test_a_frame_carries_only_the_fields_the_model_declares(tree_wired: Any) -> None:
    """The wildcard target means the payload itself is the boundary.

    `'*'` is the only target that can reach an opaque origin, so the host cannot narrow
    WHO receives a frame -- it can only narrow WHAT is in one. Every key of every frame
    must therefore be a field the package's own model declares, and a fold's other
    members must never ride along.
    """
    await _call(md.api_member_dashboard, _request(tree_wired.app, "GET", TREE_PATH))
    allowed = set(dp.subscribed_fields(dp.parse_package(tree_wired.store.content)))
    tree_wired.frames.clear()
    _publish(
        "work",
        {"items": [{"id": "it_1"}], "conductor": {"goal": "secret goal"}, "omitted": 4},
        1,
    )
    assert tree_wired.frames, "the fold moved and nothing was sent"
    for _kind, data in tree_wired.frames:
        extra = set(data["read"]["fields"]) - allowed
        assert not extra, f"a frame carried {sorted(extra)}, which the model does not declare"
    sent = json.dumps([d for _k, d in tree_wired.frames])
    assert "secret goal" not in sent, "a fold member the model omits reached the wire"


def test_the_tab_ignores_a_message_from_any_other_window() -> None:
    """The other half of the wildcard's safety: the host accepts nothing back from a
    window that is not this frame's own.

    Every frame the dashboard embeds runs with `allow-scripts`, so a sibling can reach
    `parent.frames[i]` and post. The tab's listener compares `event.source` against the
    frame's `contentWindow` before reading anything, and the frame's own bootstrap makes
    the mirror-image check against `parent`.
    """
    tab = (
        Path(mdf.__file__).resolve().parents[4]
        / "website"
        / "src"
        / "pages"
        / "members"
        / "CrewDynamicDashboard.tsx"
    )
    if not tab.exists():  # pragma: no cover - a source checkout always has it
        pytest.skip("the website source is not in this build")
    text = tab.read_text(encoding="utf-8")
    assert (
        "event.source !== win" in text
    ), "the tab reads a posted message without checking which window sent it"
    # One suppression per call, each carrying the reason -- not a file-level disable,
    # which would also cover a call added later for a different window.
    assert text.count("nosemgrep: javascript.browser.security.wildcard-postmessage") == 2
    # The DECLARATION, not the file's prose. The file explains at length that it
    # withholds same-origin access, so a text search for that token matches the
    # explanation and passes whatever the sandbox actually grants.
    assert "export const CREW_DASHBOARD_SANDBOX = 'allow-scripts'" in text, (
        "the frame's sandbox is no longer exactly allow-scripts; same-origin access "
        "would give the page the dashboard's own storage and cookies, and a non-opaque "
        "origin would also make the wildcard postMessage target a real choice"
    )


# --------------------------------------------------------------------------- #
# The two fenced findings on 6ec89205e5. Each asserts the ATTACK FAILS, not
# merely that a guard exists, because a guard that is present and reachable
# around is the shape both of these had.
# --------------------------------------------------------------------------- #


class _Linked:
    """An artifact whose bytes come from a FILE: a live source pointer.

    `source_path` set with `source_copy_only` false is the store's own spelling for
    "reads serve the file's current bytes". Nothing in the artifact API gates those
    bytes, so the write gate never sees them.
    """

    def __init__(self, content: str, *, kind: str = "dashboard") -> None:
        self.slug = PACKAGE_SLUG
        self.content = content
        self.version = 2
        self.kind = kind
        self.source_path = "/tmp/anywhere/page.json"
        self.source_root = "/tmp/anywhere"
        self.source_copy_only = False


@pytest.mark.asyncio
async def test_a_live_backed_artifact_is_not_armed_even_bound_to_this_crewmate(
    wired: Any, monkeypatch: pytest.MonkeyPatch
) -> None:
    """THE ATTACK: a worker edits the FILE behind a linked artifact into a package bound
    to this crewmate, then the owner opens the route.

    The write gate cannot have seen those bytes -- they were never submitted through the
    artifact API -- so if the route parses and arms whatever the store returns, it runs
    worker-authored script against this crewmate's own fold values. The refusal must
    land before the parse.
    """
    linked = _Linked(dp.dump_package(_package()))
    monkeypatch.setattr(
        "kiro_crew.artifacts.get_default_store", lambda: SimpleNamespace(get=lambda slug: linked)
    )

    def _never(*_a: Any, **_k: Any) -> Any:
        raise AssertionError("a live-backed artifact reached the package parser")

    monkeypatch.setattr("kiro_crew.artifact_store.dashboard_package.parse_package", _never)
    status, body = await _call(md.api_member_dashboard, _request(wired.app, "GET", PATH))
    assert status == 409
    assert body["code"] == "dashboard_live_source"
    assert not wired.app.get("member_dashboard_live_pages"), "a page was armed anyway"


@pytest.mark.asyncio
async def test_an_artifact_of_another_kind_is_not_parsed_as_a_package(
    wired: Any, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The record's own kind decides, not the bytes. A document whose text happens to
    parse as a package is still not a dashboard."""
    note = SimpleNamespace(
        slug=PACKAGE_SLUG, content=dp.dump_package(_package()), version=1, kind="note"
    )
    monkeypatch.setattr(
        "kiro_crew.artifacts.get_default_store", lambda: SimpleNamespace(get=lambda slug: note)
    )
    status, body = await _call(md.api_member_dashboard, _request(wired.app, "GET", PATH))
    assert status == 409
    assert body["code"] == "dashboard_not_a_package"


@pytest.mark.asyncio
async def test_a_copied_artifact_with_provenance_is_still_served(wired: Any) -> None:
    """`source_copy_only` means the artifact owns its bytes and the path is provenance.
    Refusing those too would turn every promoted-then-copied package away."""
    copied = _Linked(dp.dump_package(_package()))
    copied.source_copy_only = True
    wired.store.get = lambda slug: copied  # type: ignore[method-assign]
    status, _body = await _call(md.api_member_dashboard, _request(wired.app, "GET", PATH))
    assert status == 200


def test_a_refetch_cannot_publish_a_stale_snapshot_under_a_newer_version() -> None:
    """THE ATTACK ON THE DATA, which is a lost update rather than an intruder.

    A fold advances between a refetch's value capture and the moment it takes a version.
    The patch carrying the NEWER value goes out first; the refetch then stamps the OLDER
    snapshot with the next version, and the page accepts it as the latest and silently
    goes backwards -- with no gap for it to detect, because the numbers are consecutive.

    Driven deterministically: a second thread advances the fold while the refetch is
    inside its own read, and the assertion is on the ORDER of what was sent.
    """
    import threading

    frames: list[dict[str, Any]] = []
    live = mdf.LivePage(SLUG, SLUG, _package(), frames.append)
    live.arm()
    try:
        _publish("work", {"conductor": {"goal": "first"}}, 1)
        inside = threading.Event()
        released = threading.Event()
        real_values = live._values

        def _slow_values() -> Any:
            # Signal that the read has begun, then wait: this is the window the lost
            # update used. With the read inside the lock, the folder's thread blocks
            # here instead of slipping a newer value past it.
            out = real_values()
            inside.set()
            released.wait(timeout=5)
            return out

        live._values = _slow_values  # type: ignore[method-assign]

        def _advance() -> None:
            inside.wait(timeout=5)
            _publish("work", {"conductor": {"goal": "second"}}, 2)
            released.set()

        racer = threading.Thread(target=_advance)
        racer.start()
        whole = live.refetch()
        racer.join(timeout=10)
        live._values = real_values  # type: ignore[method-assign]

        # Whatever the interleaving, the page must never be told that "first" is newer
        # than "second": either the refetch carries the newer value, or its version is
        # below the patch that does.
        newest = max((f["version"] for f in frames), default=0)
        if whole["fields"]["goal"] != "second":
            assert whole["version"] < newest, (
                "the refetch stamped the older snapshot with a version above the patch "
                "carrying the newer one, so the page accepts it and goes backwards"
            )
    finally:
        live.disarm()
