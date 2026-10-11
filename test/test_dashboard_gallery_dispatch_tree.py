"""The gallery's dispatch-tree page: the first thing that consumes the tree fold.

``read_tree_projection`` shipped GATED -- "no production caller reaches it yet ... this
read ships with the first dashboard consumer that can bound the walk". This page is that
consumer, so the claims worth pinning are the ones that would make the gate a lie:

* the page is a package the write gate accepts, so an owner can copy it and it will save;
* it reads the tree fold by its real field names, so the fields resolve rather than
  rendering a tree of em dashes;
* it is served through ``dashboard_frame``, bindings and all;
* a value holding markup stays text on the page, which is the one property a reference
  page must not teach an author to break.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest

from kiro_crew.artifact_store import dashboard_package as dp
from kiro_crew.dashboard.handlers import member_dashboard_fields as mdf

GALLERY = (
    Path(mdf.__file__).resolve().parents[2]
    / "builtin_skills"
    / "dashboard"
    / "gallery"
    / "dispatch-tree"
    / "package.json"
)


@pytest.fixture(scope="module")
def page() -> dict[str, Any]:
    """The stored gallery package, through the same gate a save would use."""
    return dp.parse_package(GALLERY.read_text(encoding="utf-8"))


def test_the_gallery_page_is_a_package_the_write_gate_accepts(page: dict[str, Any]) -> None:
    """A reference page an owner cannot save is worse than no reference page: they copy
    it, the save is refused, and the refusal reads as their mistake."""
    assert page["view"]["html"]
    assert dp.canonical_package_content(json.dumps(page))


def test_it_subscribes_to_the_tree_fold_and_the_work_ledger(page: dict[str, Any]) -> None:
    sources = dp.subscribed_fields(page)
    assert sources["boards"] == "fold:worktree.boards"
    assert sources["roots"] == "fold:worktree.roots"
    assert sources["work_items"] == "fold:work.items"


def test_every_path_it_binds_is_a_field_the_fold_really_carries(page: dict[str, Any]) -> None:
    """Pinned against the fold's own declared field list, not against what the page
    expects. A page naming a key the fold does not carry resolves to nothing, and the
    page then draws an empty tree over a log that has the boards in it."""
    catalog = json.loads(
        (GALLERY.parents[3] / "kirocrew-dev" / "dashboard-template" / "folds.json").read_text(
            encoding="utf-8"
        )
    )
    folds = catalog["folds"] if isinstance(catalog, dict) else catalog
    carried = {
        entry["name"]: {field["name"] for field in entry.get("fields", [])} for entry in folds
    }
    for name, spec in (page["model"]["types"]).items():
        source = spec["source"]
        if source.get("agentic"):
            continue
        head = source["path"].split(".")[0]
        assert head in carried[source["fold"]], (
            f"field {name!r} reads {source['fold']}.{source['path']}, and the "
            f"{source['fold']} fold carries no {head!r}"
        )


def test_the_tree_fold_is_the_one_with_no_bus_scope(page: dict[str, Any]) -> None:
    """Which is why this page needs a read rather than a subscription: a tree joins the
    logs of many slots, so its value is stale when any member's log grows and there is
    no key an event could carry to say which."""
    from kiro_crew.crew_log.projection import TREE_PROJECTION_NAMES
    from kiro_crew.dashboard_feed import scope_for

    assert "worktree" in TREE_PROJECTION_NAMES
    assert scope_for("worktree") == ""


def test_the_page_draws_from_subscribe_rather_than_polling(page: dict[str, Any]) -> None:
    """The reason the page updates live at all. A reference page that polled would teach
    every copy of it to poll."""
    html = page["view"]["html"]
    assert "window.kirocrew.subscribe('boards'" in html
    assert "setInterval" not in html and "fetch(" not in html


def test_the_page_builds_its_rows_as_text_never_as_markup(page: dict[str, Any]) -> None:
    """A board's goal is whatever a conductor wrote in the crew log. The reference page
    is the thing authors copy, so it has to show the safe spelling: elements created and
    ``textContent`` set, never a template string assigned to ``innerHTML``."""
    html = page["view"]["html"]
    assert "textContent" in html
    for sink in ("innerHTML", "outerHTML", "insertAdjacentHTML", "document.write"):
        assert sink not in html, f"the gallery page reaches {sink}"


def test_it_is_served_through_the_frame_contract(page: dict[str, Any]) -> None:
    from kiro_crew import dashboard_frame

    read = dashboard_frame.read_payload({"roots": 2, "cycles": 0, "unresolved": 0})
    body = mdf.render_package_page(page, read)
    assert page["view"]["html"] in body, "the page is not placed verbatim"
    assert dashboard_frame.BINDING_ATTRIBUTE in body
    assert "kirocrew-dashboard-data" in body
    assert body.index("kirocrew-dashboard-data") < body.index(
        page["view"]["html"]
    ), "the data island follows the page, so the page's own first script reads undefined"


def test_a_board_goal_holding_markup_is_not_rendered_as_markup() -> None:
    """End to end through the host's half: a value the log recorded reaches the document
    escaped, so a goal carrying a tag is text in the island and text in the cell."""
    from kiro_crew import dashboard_frame

    read = dashboard_frame.read_payload({"roots": "<img src=x onerror=alert(1)>"})
    body = dashboard_frame.compose_body('<p data-dashboard-field="roots"></p>', read)
    assert "<img src=x" not in body
    assert "\\u003cimg src=x" in body


def test_the_gallery_page_fits_the_view_cap(page: dict[str, Any]) -> None:
    assert len(page["view"]["html"].encode("utf-8")) < dp.MAX_VIEW_HTML_BYTES
