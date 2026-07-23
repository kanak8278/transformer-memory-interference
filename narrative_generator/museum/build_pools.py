#!/usr/bin/env python3
"""Build the museum domain value pools.

Design rationale (see 36_museum_ATEMPORAL_DESIGN.md §5.1):

  Artwork titles are drawn from the SAME pool plain KV uses
  (mechanistic_probing_v2/core/data/arbitrary_single.json), capitalized and
  filtered for substring safety. Reusing plain's pool means museum and plain
  values share an identical distribution — same token lengths, same word
  frequencies — so a museum-vs-plain difference cannot be attributed to the
  values. Only the sentence frame differs, which is the variable under test.

  The titles read naturally because the domain is framed as a CONTEMPORARY ART
  museum, where single abstract words (Ripple, Onset, Bliss, Pardon, Recess)
  are idiomatic titling.

Substring safety is required because the behavioural scorer at
v3/scripts/experiments/narrative_new_domains.py:88-130 grades with
case-insensitive substring containment. If one value contains another, a wrong
answer can be graded correct.

Run:  python3 narrative_generator/museum/build_pools.py
"""
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
PLAIN_POOL = ROOT / "mechanistic_probing_v2" / "core" / "data" / "arbitrary_single.json"
OUT = Path(__file__).resolve().parent / "data" / "museum_data.json"


# ── visitor names ────────────────────────────────────────────────────────────
# Deliberately multi-origin. Sized so a 15-visitor trial draws a varied roster
# across seeds. These double as VALUES when `companion` is the tracked
# attribute, so they are held to the same substring rules as artwork titles.
VISITOR_CANDIDATES = [
    "Rohan", "Priya", "Marcus", "Ingrid", "Tomas", "Nadia", "Felix", "Ayesha",
    "Diego", "Lena", "Omar", "Clara", "Yusuf", "Mira", "Anton", "Farah",
    "Elias", "Rina", "Hugo", "Sofia", "Kwame", "Ilse", "Rafael", "Noor",
    "Bjorn", "Amara", "Viktor", "Leila", "Pedro", "Astrid", "Hassan", "Greta",
    "Nikhil", "Rosa", "Emeka", "Sanne", "Dmitri", "Yuki", "Caleb", "Ines",
    "Tariq", "Maja", "Andres", "Zara", "Lukas", "Fatima", "Sergei", "Nadja",
    "Arun", "Beatriz", "Kofi", "Elsa", "Ravi", "Marta", "Idris", "Hanna",
    "Pablo", "Nour", "Stefan", "Aisha", "Milos", "Carmen", "Jomo", "Britta",
    "Vikram", "Lucia", "Samir", "Freya", "Nina", "Bashir", "Alma",
    "Tenzin", "Paloma", "Ewan", "Suri", "Mateo", "Runa", "Hakim", "Dagny",
    "Sanjay", "Lorena", "Musa", "Kirsi", "Enzo", "Amina", "Casper", "Delia",
    "Rustam", "Manon", "Otto", "Sabine", "Jae", "Rania", "Bruno", "Katri",
    "Nabil", "Elodie", "Kiran", "Solveig", "Ammar", "Renata", "Sven", "Dilara",
    "Pavel", "Naima", "Emil", "Bianca", "Tomasz", "Hina", "Aleks", "Marisol",
    "Ibrahim", "Anneke", "Rohit", "Pilar", "Zoltan", "Maren", "Cyrus", "Ondine",
    "Halim", "Franka", "Devika", "Janos", "Rosalind", "Teodor", "Nuria", "Wael",
    "Katinka", "Bodhi", "Salma", "Ferran", "Ulrika", "Nizar", "Camila",
    "Anselm", "Havva", "Gustav", "Malika", "Piotr", "Serena", "Adnan", "Tove",
    "Rasmus", "Yasmin", "Lars", "Chandra", "Nikolai", "Esme", "Baptiste",
    "Roshni", "Emeric", "Sirpa", "Karim", "Valeria", "Jonas", "Anouk",
]

DOCENTS = [
    "Whitfield", "Arbogast", "Pellerin", "Vosburgh", "Iniesta", "Radcliffe",
    "Okonkwo", "Sandoval", "Brightmore", "Kasprzak", "Yamashiro",
    "Trelawny", "Bergstrom", "Nakagawa", "Fontaine", "Halloran", "Zimmerli",
    "Castellan", "Duplessis", "Oyelaran",
]

GALLERIES = [
    "Ellery Room", "Kessler Gallery", "Marchetti Room", "Vance Room",
    "Osgood Gallery", "Brightwater Room", "Calloway Gallery", "Thorne Room",
    "Ambrose Gallery", "Lindqvist Room", "Ferraro Room", "Whitlock Gallery",
    "Danforth Room", "Mercier Gallery", "Ashgrove Room",
]

WINGS = [
    "Pemberton Wing", "Ashcombe Wing", "Halvard Wing", "Restrepo Wing",
    "Nyborg Wing", "Tanaka Wing", "Loverin Wing", "Escalante Wing",
]

EXHIBITIONS = [
    "Interval Studies", "Ground and Figure", "The Quiet Register",
    "Material Evidence", "Nine Rooms", "Slow Light", "After the Grid",
    "Provisional Forms", "Held Breath", "The Long Table", "Salt and Paper",
    "Second Sight",
]

AUDIO_TOPICS = [
    "Colour and Ground", "The Studio Years", "Provenance Notes",
    "Sitters and Subjects", "Working Methods", "The Late Canvases",
    "Frames and Framing", "Conservation Findings", "Patrons and Commissions",
    "Drawing Practice",
]

LANGUAGES = [
    "English", "Japanese", "Portuguese", "Arabic", "Korean", "Dutch",
    "Hungarian", "Swahili", "Catalan", "Finnish",
]

SEATING = [
    "Larkspur Bench", "Ives Alcove", "Redgrave Bench", "Sorrel Alcove",
    "Petrie Bench", "Hollowell Alcove", "Grantley Bench", "Maskell Alcove",
]

INTERPRETIVE_PANELS = [
    "Making the Pigment", "Who Commissioned This", "A Note on Scale",
    "Restoration History", "The Artist's Circle", "Reading the Inscription",
    "Studio Assistants", "Support and Ground",
]

TALKS = [
    "Fifteen Minutes With", "Curator's Choice", "Close Looking",
    "Question Hour", "Behind the Label", "Sketchbook Session",
    "Materials Table", "Open Floor",
]

MEDIA = [
    "bronzes", "textiles", "works on paper", "ceramics", "photographs",
    "panel paintings", "glass", "lacquerware", "prints", "plaster casts",
]

GUIDEBOOK_SECTIONS = [
    "Plan of the Building", "Highlights", "Recent Acquisitions",
    "Notes on the Collection", "Visiting with Children", "Access Information",
]

CONSERVATION_STATIONS = [
    "Varnish Table", "Paper Bench", "Frame Shop", "Textile Station",
    "Imaging Bay", "Mount Making",
]

SHOP_ITEMS = [
    "postcard rack", "exhibition catalogue", "pigment set", "linen tote",
    "print folio", "enamel pin", "notebook stack", "poster tube",
]

MUSEUMS = [
    ("Pelham Museum of Art", "west building"),
    ("Ostrander Collection", "north building"),
    ("Carrow Museum of Modern Art", "main building"),
    ("The Vance Institute", "east building"),
    ("Halloway Museum", "annexe"),
]


def substring_free(items):
    """Drop any item that is a case-insensitive substring of another."""
    low = [i.lower() for i in items]
    bad = {i for i, w in zip(items, low)
           if any(w != o and w in o for o in low)}
    return [i for i in items if i not in bad]


def cross_safe(primary, secondary):
    """Drop items from `secondary` that collide with anything in `primary`."""
    plow = [p.lower() for p in primary]
    out = []
    for s in secondary:
        sl = s.lower()
        if any(sl in p or p in sl for p in plow):
            continue
        out.append(s)
    return out


def build_artworks():
    values = json.loads(PLAIN_POOL.read_text())["values"]
    titles = [w.capitalize() for w in values]
    return substring_free(titles)


def main():
    artworks = build_artworks()
    visitors = substring_free(VISITOR_CANDIDATES)
    visitors = cross_safe(artworks, visitors)
    docents = cross_safe(artworks, substring_free(DOCENTS))

    data = {
        "metadata": {
            "domain": "museum",
            "artwork_source": "arbitrary_single.json (plain KV pool), capitalized",
            "artwork_rationale": (
                "Identical value distribution to plain KV so museum-vs-plain "
                "cannot differ by value pool; only the sentence frame differs."
            ),
            "framing": "contemporary art museum — single abstract-word titles are idiomatic",
            "substring_rule": (
                "no value is a case-insensitive substring of another, within or "
                "across the ARTWORKS and visitor/docent pools; required by the "
                "containment scorer in narrative_new_domains.py:88-130"
            ),
        },
        "ARTWORKS": artworks,
        "VISITOR_NAMES": visitors,
        "DOCENTS": docents,
        "GALLERIES": GALLERIES,
        "WINGS": WINGS,
        "EXHIBITIONS": EXHIBITIONS,
        "AUDIO_TOPICS": AUDIO_TOPICS,
        "LANGUAGES": LANGUAGES,
        "SEATING": SEATING,
        "INTERPRETIVE_PANELS": INTERPRETIVE_PANELS,
        "TALKS": TALKS,
        "MEDIA": MEDIA,
        "GUIDEBOOK_SECTIONS": GUIDEBOOK_SECTIONS,
        "CONSERVATION_STATIONS": CONSERVATION_STATIONS,
        "SHOP_ITEMS": SHOP_ITEMS,
        "MUSEUMS": MUSEUMS,
    }

    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(data, indent=2, ensure_ascii=False))

    print(f"wrote {OUT}")
    print(f"  ARTWORKS      : {len(artworks)}")
    print(f"  VISITOR_NAMES : {len(visitors)}")
    print(f"  DOCENTS       : {len(docents)}")
    for k in ("GALLERIES", "WINGS", "EXHIBITIONS", "AUDIO_TOPICS", "SEATING"):
        print(f"  {k:<14}: {len(data[k])}")


if __name__ == "__main__":
    main()
