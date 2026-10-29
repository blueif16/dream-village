"""memsys.store: the per-principal memory directory (Claude apps layout) and its code-enforced rules.

Every rule here is one the model meets as an error message, so each test checks both that the bad edit is refused
and that the file on disk is untouched.
"""

from __future__ import annotations

import pytest

from memsys.store import Store, StoreError


def topic(name: str, description: str = "Food tastes and routines", aliases: str = "", body: str = "- [stated] x") -> str:
    return f"---\nname: {name}\ndescription: {description}\nsources: [chat]\naliases: [{aliases}]\n---\n\n{body}\n"


@pytest.fixture
def store(tmp_path) -> Store:
    return Store(tmp_path / "kai")


def version_of(store: Store, path: str) -> str:
    (got,) = store.read([path])
    return got["version"]


# ----------------------------------------------------------------------------- read


def test_read_returns_several_files_with_content_and_version_in_one_call(store):
    store.write("/topics/food.md", topic("food"))
    store.write("/people/dani.md", topic("dani", "Dani Cho, roommate"))
    got = store.read(["/topics/food.md", "/people/dani.md"])
    assert [g["path"] for g in got] == ["/topics/food.md", "/people/dani.md"]
    assert got[1]["content"] == topic("dani", "Dani Cho, roommate")
    assert got[0]["version"] and got[0]["version"] != got[1]["version"]


def test_read_of_a_missing_file_is_a_per_path_error_not_an_exception(store):
    store.write("/topics/food.md", topic("food"))
    got = store.read(["/topics/nope.md", "/topics/food.md"])
    assert "error" in got[0] and "content" not in got[0]
    assert got[1]["content"] == topic("food")


def test_version_token_changes_when_content_changes(store):
    store.write("/topics/food.md", topic("food"))
    v1 = version_of(store, "/topics/food.md")
    store.append("/topics/food.md", "- [stated] tea now (previously coffee)")
    assert version_of(store, "/topics/food.md") != v1


# ----------------------------------------------------------------------------- str_replace


def test_str_replace_with_a_unique_match_edits_the_file(store):
    store.write("/topics/food.md", topic("food", body="- [stated] drinks coffee"))
    store.str_replace("/topics/food.md", "drinks coffee", "drinks tea (previously coffee)", version_of(store, "/topics/food.md"))
    assert "- [stated] drinks tea (previously coffee)" in store.text("/topics/food.md")
    assert "drinks coffee\n" not in store.text("/topics/food.md")


def test_str_replace_with_an_ambiguous_match_is_refused_and_leaves_the_file(store):
    store.write("/topics/food.md", topic("food", body="- [stated] coffee\n- [stated] coffee"))
    before = store.text("/topics/food.md")
    with pytest.raises(StoreError, match="2"):
        store.str_replace("/topics/food.md", "coffee", "tea", version_of(store, "/topics/food.md"))
    assert store.text("/topics/food.md") == before


def test_str_replace_with_no_match_is_refused_and_leaves_the_file(store):
    store.write("/topics/food.md", topic("food", body="- [stated] coffee"))
    before = store.text("/topics/food.md")
    with pytest.raises(StoreError, match="not found"):
        store.str_replace("/topics/food.md", "matcha", "tea", version_of(store, "/topics/food.md"))
    assert store.text("/topics/food.md") == before


# ----------------------------------------------------------------------------- version tokens (read before write)


def test_str_replace_with_a_stale_version_is_refused(store):
    store.write("/topics/food.md", topic("food", body="- [stated] coffee"))
    stale = version_of(store, "/topics/food.md")
    store.append("/topics/food.md", "- [stated] oat milk")
    before = store.text("/topics/food.md")
    with pytest.raises(StoreError, match="(?i)stale|version"):
        store.str_replace("/topics/food.md", "coffee", "tea", stale)
    assert store.text("/topics/food.md") == before


def test_overwriting_an_existing_file_needs_its_current_version(store):
    store.write("/topics/food.md", topic("food"))
    with pytest.raises(StoreError, match="(?i)version"):
        store.write("/topics/food.md", topic("food", body="- [stated] rewritten"))
    with pytest.raises(StoreError, match="(?i)stale|version"):
        store.write("/topics/food.md", topic("food", body="- [stated] rewritten"), version="0" * 12)
    assert "rewritten" not in store.text("/topics/food.md")
    store.write("/topics/food.md", topic("food", body="- [stated] rewritten"), version=version_of(store, "/topics/food.md"))
    assert "rewritten" in store.text("/topics/food.md")


def test_each_edit_returns_the_new_version_so_edits_can_chain(store):
    v = store.write("/topics/food.md", topic("food", body="- [stated] coffee"))
    v = store.str_replace("/topics/food.md", "coffee", "tea", v)
    store.str_replace("/topics/food.md", "tea", "matcha", v)
    assert "- [stated] matcha" in store.text("/topics/food.md")


# ----------------------------------------------------------------------------- profile cap, description, format


def test_profile_of_300_words_is_accepted_and_301_is_refused_with_the_reason(store):
    store.write("/profile.md", " ".join(["word"] * 300))
    v = version_of(store, "/profile.md")
    with pytest.raises(StoreError, match="300"):
        store.write("/profile.md", " ".join(["word"] * 301), version=v)
    assert len(store.text("/profile.md").split()) == 300


def test_profile_cap_also_holds_for_append(store):
    store.write("/profile.md", " ".join(["word"] * 299))
    with pytest.raises(StoreError, match="300"):
        store.append("/profile.md", "- two words")
    assert len(store.text("/profile.md").split()) == 299


def test_description_must_be_under_150_characters(store):
    store.write("/topics/a.md", topic("a", "d" * 149))
    with pytest.raises(StoreError, match="150"):
        store.write("/topics/b.md", topic("b", "d" * 150))
    assert not store.exists("/topics/b.md")


def test_a_memory_file_needs_frontmatter_whose_name_is_its_path_stem(store):
    with pytest.raises(StoreError, match="(?i)frontmatter"):
        store.write("/topics/food.md", "- [stated] coffee\n")
    with pytest.raises(StoreError, match="food"):
        store.write("/topics/food.md", topic("drinks"))
    assert not store.exists("/topics/food.md")


def test_paths_outside_the_layout_are_refused(store, tmp_path):
    for bad in ("/../escape.md", "/notes/x.md", "/topics/Food.md", "/topics/sub/x.md", "/topics/x.txt"):
        with pytest.raises(StoreError, match="(?i)path"):
            store.write(bad, topic("x"))
    assert not (tmp_path / "escape.md").exists()


# ----------------------------------------------------------------------------- one file per subject


def test_creating_a_file_whose_name_is_an_existing_alias_is_refused(store):
    store.write("/people/mom.md", topic("mom", "Their mother, Rosa", aliases="rosa"))
    with pytest.raises(StoreError, match="/people/mom.md"):
        store.write("/people/rosa.md", topic("rosa", "Rosa"))
    assert not store.exists("/people/rosa.md")


def test_creating_a_file_whose_alias_is_an_existing_name_is_refused_ignoring_case_and_dashes(store):
    store.write("/areas/apartment-search.md", topic("apartment-search", "The apartment search"))
    with pytest.raises(StoreError, match="/areas/apartment-search.md"):
        store.write("/topics/housing.md", topic("housing", "Housing", aliases="Apartment Search"))
    assert not store.exists("/topics/housing.md")


def test_an_alias_added_by_edit_may_not_collide_with_another_file(store):
    store.write("/people/mom.md", topic("mom", "Their mother", aliases="rosa"))
    v = store.write("/people/dani.md", topic("dani", "Roommate", aliases="d"))
    with pytest.raises(StoreError, match="/people/mom.md"):
        store.str_replace("/people/dani.md", "aliases: [d]", "aliases: [d, Rosa]", v)


# ----------------------------------------------------------------------------- append, delete, listing


def test_append_adds_one_line_and_refuses_several(store):
    store.write("/topics/food.md", topic("food", body="- [stated] coffee"))
    store.append("/topics/food.md", "- [stated] oat milk")
    assert store.text("/topics/food.md").endswith("- [stated] coffee\n- [stated] oat milk\n")
    with pytest.raises(StoreError, match="(?i)one line"):
        store.append("/topics/food.md", "- a\n- b")


def test_delete_removes_the_file(store):
    store.write("/topics/food.md", topic("food"))
    store.delete("/topics/food.md")
    assert not store.exists("/topics/food.md")
    with pytest.raises(StoreError, match="(?i)not found"):
        store.delete("/topics/food.md")


def test_listing_renders_every_memory_file_but_profile_and_preferences(store):
    store.write("/profile.md", "Kai, 19, community college student.")
    store.write("/preferences.md", "Short answers.")
    store.write("/topics/food.md", topic("food", "Food tastes; names [[dani]]"))
    store.write("/people/mom.md", topic("mom", "Their mother, Rosa", aliases="rosa, Ma"))
    listing = store.listing()
    assert listing.startswith("<memory_listing>") and listing.rstrip().endswith("</memory_listing>")
    lines = [ln for ln in listing.splitlines() if ln.startswith("- ")]
    assert len(lines) == 2
    assert lines[0].startswith("- /people/mom.md") and lines[1].startswith("- /topics/food.md")  # sorted by path
    assert "Their mother, Rosa" in lines[0] and "aliases: [rosa, Ma]" in lines[0] and "sources: [chat]" in lines[0]
    assert "profile" not in listing and "preferences" not in listing


def test_listing_of_an_empty_store_says_so(store):
    assert "(no memory files yet)" in store.listing()
