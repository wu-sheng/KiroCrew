"""A dashboard package is written by the crewmate it is bound to, and by nobody else.

The split this protects is the one that makes a free-HTML page safe to run at all: the
owner composes the page and a person approves it, workers only `work_report` and the fold
carries their reports to it. A worker that could write the package could write the page --
and the page runs script inside the frame that holds the owner's own fold values.

The refusal is a SENTENCE because the party that reads it is an agent that can act on it:
whose dashboard this is, and what to do instead. A code would be rendered to a worker as
`owner_only` with nothing to do next.
"""

from __future__ import annotations

from typing import Any

import pytest

from kiro_crew.artifact_store import dashboard_package as dp


@pytest.fixture()
def roster(monkeypatch: pytest.MonkeyPatch) -> None:
    """Slugs for two crews, resolved the way the roster resolves them."""
    import kiro_crew.members as members_mod

    slugs = {"Atlas": "atlas", "Nyx": "nyx"}

    def member_slug(name: str, config: Any = None) -> str:
        if name not in slugs:
            raise members_mod.MemberSlugError(name)
        return slugs[name]

    monkeypatch.setattr(members_mod, "member_slug", member_slug)


class TestACrewmateBinding:
    def test_the_owner_may_write_its_own_package(self, roster: None) -> None:
        assert dp.binding_refusal("crewmate:atlas", crew="Atlas") == ""

    def test_a_worker_is_refused_and_told_what_to_do(self, roster: None) -> None:
        said = dp.binding_refusal("crewmate:atlas", crew="Nyx")
        assert "atlas" in said
        assert "Nyx" in said
        assert "work_report" in said, "the refusal does not say what the worker should do"

    def test_a_caller_with_no_crew_is_refused(self, roster: None) -> None:
        """A gate whose default is "no identity, no objection" is not a gate."""
        assert dp.binding_refusal("crewmate:atlas") != ""

    def test_a_caller_whose_name_has_no_slug_is_refused(self, roster: None) -> None:
        """An unslugifiable caller cannot be shown to own anything, which is the same
        answer as a mismatch rather than a crash on the write path."""
        assert dp.binding_refusal("crewmate:atlas", crew="Ghost") != ""

    def test_the_slug_is_compared_not_the_display_name(self, roster: None) -> None:
        """Slugification is lossy, so two configured names can reach one slug. Comparing
        names would let each pass for the other's page; comparing slugs is what the
        binding actually stores."""
        assert dp.binding_refusal("crewmate:atlas", crew="atlas") != ""


class TestASessionBinding:
    def test_the_bound_session_may_write(self) -> None:
        assert dp.binding_refusal("session:chat-7-17910", slot="chat-7-17910") == ""

    def test_another_session_is_refused(self) -> None:
        said = dp.binding_refusal("session:chat-7-17910", slot="chat-9-17911")
        assert "chat-7-17910" in said
        assert "work_report" in said

    def test_a_caller_with_no_session_is_refused(self) -> None:
        assert dp.binding_refusal("session:chat-7-17910") != ""

    def test_a_crew_identity_does_not_satisfy_a_session_binding(self, roster: None) -> None:
        """The two scopes name different things. A crewmate that happened to pass here
        would be writing a page bound to a slot it does not own."""
        assert dp.binding_refusal("session:chat-7-17910", crew="Atlas") != ""


class TestAMalformedBinding:
    def test_a_binding_that_is_not_one_is_refused_with_its_own_reason(self) -> None:
        """Refused as a malformed binding rather than as a wrong owner: an author that
        wrote the grammar wrong needs the grammar, not a sentence about ownership."""
        for bad in ("atlas", "", None, 7, "crewmate:", "session:dashboard:chat-7"):
            said = dp.binding_refusal(bad, crew="Atlas", slot="chat-7-17910")
            assert said != ""
            assert "work_report" not in said, f"{bad!r} was answered as an ownership refusal"


# --------------------------------------------------------------------------- #
# Two ways the gate was skippable. Both were reported by review on
# e1f2ce06cb and both are authorization bypasses, so each gets a test that
# fails without the fix rather than a comment saying it is handled.
# --------------------------------------------------------------------------- #


class _Req:
    """The two things the gate reads off a request, and nothing else."""

    def __init__(self, secret: bool = True) -> None:
        self.headers = {"X-Internal-Secret": "s"} if secret else {}


@pytest.fixture()
def worker(monkeypatch: pytest.MonkeyPatch, roster: None) -> None:
    """A caller that is the crew ``Nyx`` -- not ``atlas``, whose pages these are."""
    from kiro_crew.dashboard.handlers import member_dashboard_fields as mdf

    monkeypatch.setattr(mdf, "writing_caller", lambda request: ("Nyx", "chat-9-17911"))


def _package_text(bound_to: str = "crewmate:atlas", kind: str = '"dashboard"') -> str:
    """A valid package as submitted BYTES, so a test can spell ``kind`` its own way."""
    return (
        "{"
        f'"kind":{kind},'
        f'"bound_to":"{bound_to}",'
        '"model":{"types":{"goal":{"type":"text",'
        '"source":{"fold":"work","path":"conductor.goal"}}}},'
        # A page that BINDS NOTHING, drawing from `subscribe` instead. The binding
        # attribute is itself spelled `data-dashboard-field`, so a page that used one
        # would put the word in the bytes and the old substring check would have caught
        # this payload by accident -- which is not the payload an attacker sends.
        '"view":{"html":"<canvas id=t></canvas>"},'
        '"theme":{"tokens":{}}'
        "}"
    )


class TestTheKindCannotBeHiddenFromTheGate:
    """The gate classified the content by looking for the word ``dashboard`` in the
    submitted bytes. JSON can spell any character as an escape, so the word need never
    appear -- and the store's own ``json.loads`` decodes it to a package regardless.

    Reported independently by two review lanes, which is the signal that it is reachable
    rather than theoretical: the worker needs no special access, only a text editor.
    """

    def test_a_package_whose_kind_is_escaped_is_still_gated(self, worker: None) -> None:
        from kiro_crew.dashboard.handlers import member_dashboard_fields as mdf

        # No literal "dashboard" anywhere in these bytes.
        text = _package_text(kind='"dash\\u0062oard"')
        assert "dashboard" not in text
        said = mdf.package_write_refusal(_Req(), text)
        assert said, "a worker can write another crewmate's page by escaping one letter"
        assert "atlas" in said

    def test_every_letter_escaped_is_still_gated(self, worker: None) -> None:
        """The fix cannot be a longer list of spellings to look for."""
        from kiro_crew.dashboard.handlers import member_dashboard_fields as mdf

        escaped = "".join(f"\\u{ord(c):04x}" for c in "dashboard")
        assert mdf.package_write_refusal(_Req(), _package_text(kind=f'"{escaped}"'))

    def test_a_prose_document_is_still_not_gated(self, worker: None) -> None:
        """The cheap pre-filter has to stay cheap AND unescapable: a JSON document
        begins with '{' and no escape hides that, so prose never reaches a parse."""
        from kiro_crew.dashboard.handlers import member_dashboard_fields as mdf

        for text in ("# just a note", "dashboard is a word I used in prose", "", None):
            assert mdf.package_write_refusal(_Req(), text) == ""

    def test_json_that_is_not_a_package_is_not_gated(self, worker: None) -> None:
        from kiro_crew.dashboard.handlers import member_dashboard_fields as mdf

        assert mdf.package_write_refusal(_Req(), '{"kind":"note","body":"hi"}') == ""


class TestTheStoredBindingIsAuthorizedToo:
    """The gate read only the binding being WRITTEN, which is a claim the caller wrote.

    So a worker submitted a package bound to ITSELF, passed, and overwrote the slug
    holding another crewmate's page -- rebinding that crewmate's dashboard to the worker
    in one write. The binding that decides whether a slug may be written is the one the
    slug holds now.
    """

    def test_a_worker_cannot_overwrite_another_crewmates_slug_by_rebinding_it(
        self, worker: None
    ) -> None:
        from kiro_crew.dashboard.handlers import member_dashboard_fields as mdf

        said = mdf.package_write_refusal(
            _Req(),
            # Bound to the worker itself: authorizing this alone always passes.
            _package_text(bound_to="crewmate:nyx"),
            # What the slug holds now.
            _package_text(bound_to="crewmate:atlas"),
        )
        assert said, "a worker can take over another crewmate's dashboard by rebinding it"
        assert "atlas" in said

    def test_a_worker_cannot_blank_another_crewmates_page_with_prose(self, worker: None) -> None:
        """The submitted content is not a package at all, so classifying only the
        submission answers "not mine to judge" and the store overwrites the page."""
        from kiro_crew.dashboard.handlers import member_dashboard_fields as mdf

        said = mdf.package_write_refusal(_Req(), "# gone", _package_text("crewmate:atlas"))
        assert said
        assert "atlas" in said

    def test_the_owner_may_still_rewrite_its_own_page(self, worker: None) -> None:
        from kiro_crew.dashboard.handlers import member_dashboard_fields as mdf

        monkeypatched = mdf.package_write_refusal(
            _Req(), _package_text("crewmate:nyx"), _package_text("crewmate:nyx")
        )
        assert monkeypatched == ""

    def test_a_browser_caller_is_not_gated_here(self) -> None:
        """No internal secret: the person at the dashboard, already owner-gated by the
        artifact routes. Asserted so the fix above cannot quietly start refusing them."""
        from kiro_crew.dashboard.handlers import member_dashboard_fields as mdf

        assert (
            mdf.package_write_refusal(
                _Req(secret=False), _package_text(), _package_text("crewmate:atlas")
            )
            == ""
        )
