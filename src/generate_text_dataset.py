"""
generate_text_dataset.py
-------------------------
Builds a realistic, template-based civic-complaint text dataset with 5 categories:
Garbage, Road Damage, Water Leakage, Drainage Blockage, Streetlight Fault.

This programmatically combines locations, severities and phrasing templates to
create hundreds of grammatically varied, non-duplicate complaint sentences per
category, so TF-IDF + Logistic Regression learns real discriminative patterns
instead of memorizing a handful of sentences.

Run:
    python src/generate_text_dataset.py
Produces:
    data/complaints_text.csv   (columns: complaint_text, category)
"""

import csv
import random
import os

random.seed(42)

LOCATIONS = [
    "near the main market", "outside my house", "in sector 12", "behind the bus stand",
    "on MG Road", "near the school", "in the residential colony", "close to the park",
    "next to the temple", "in front of the hospital", "on the highway service road",
    "in ward number 7", "near the railway station", "behind the shopping complex",
    "in the industrial area", "on the street corner", "near the community hall",
    "in the old city area", "close to the water tank", "along the canal road",
]

SEVERITY_PHRASES = [
    "for the past three days", "since last week", "for over a month now",
    "since yesterday", "for a few hours", "repeatedly every monsoon",
    "for the last two weeks", "since this morning", "constantly",
    "on and off for days",
]

TEMPLATES = {
    "Garbage": [
        "Garbage has been piling up {loc} {sev} and nobody has come to collect it.",
        "There is an overflowing garbage bin {loc} causing a foul smell {sev}.",
        "Household waste is scattered {loc} and stray animals are spreading it around.",
        "The municipal garbage truck has not visited {loc} {sev}.",
        "Uncollected trash {loc} is attracting flies and mosquitoes {sev}.",
        "Plastic waste and food scraps are dumped {loc}, please arrange cleaning.",
        "The dustbin {loc} is broken and garbage is spilling onto the road {sev}.",
        "Residents are dumping construction debris {loc} and it needs urgent removal.",
        "Foul odor from rotting garbage {loc} is making it hard to breathe {sev}.",
        "Sanitation workers have skipped collecting waste {loc} {sev}.",
    ],
    "Road Damage": [
        "There is a huge pothole {loc} that is causing accidents {sev}.",
        "The road surface {loc} has cracked badly and needs immediate repair.",
        "A portion of the road {loc} has caved in {sev}, please send a team.",
        "Broken tar and loose gravel {loc} are damaging vehicle tyres {sev}.",
        "The footpath {loc} is uneven and dangerous for pedestrians {sev}.",
        "Heavy rains have washed away part of the road {loc} {sev}.",
        "Multiple potholes {loc} are causing traffic jams and vehicle damage.",
        "The newly laid road {loc} has already developed cracks {sev}.",
        "A speed breaker {loc} has broken down and is a hazard for two-wheelers.",
        "The road {loc} has not been repaired since the construction work {sev}.",
    ],
    "Water Leakage": [
        "A water pipeline has burst {loc} and clean water is being wasted {sev}.",
        "There is continuous water leakage from the main supply line {loc} {sev}.",
        "Drinking water is leaking onto the road {loc} causing wastage {sev}.",
        "The overhead water tank {loc} is overflowing and leaking {sev}.",
        "A broken valve {loc} is causing water to leak into nearby houses.",
        "Water supply pipes {loc} are damaged and leaking heavily {sev}.",
        "There is a major leak in the underground pipeline {loc} {sev}.",
        "Clean water keeps flowing out from a cracked pipe {loc} {sev}.",
        "The tap connection {loc} is leaking continuously and wasting water.",
        "A municipal water line {loc} has been leaking {sev} without any repair.",
    ],
    "Drainage Blockage": [
        "The drainage near {loc} is completely blocked and water is stagnating {sev}.",
        "Sewage water is overflowing from the drain {loc} {sev}.",
        "The stormwater drain {loc} is clogged with plastic and debris {sev}.",
        "Dirty drain water is entering houses {loc} due to blockage {sev}.",
        "The open drain {loc} is choked and emitting a bad smell {sev}.",
        "Rainwater is not draining properly {loc} causing waterlogging {sev}.",
        "The sewer line {loc} is blocked and overflowing onto the street.",
        "Mosquitoes are breeding in the stagnant drain water {loc} {sev}.",
        "The drainage system {loc} has collapsed and needs urgent desilting.",
        "Blocked culverts {loc} are causing flooding during rains {sev}.",
    ],
    "Streetlight Fault": [
        "The streetlight {loc} has not been working {sev} making the area unsafe at night.",
        "Several streetlights {loc} are flickering and need repair {sev}.",
        "There is no lighting {loc} at night due to a faulty streetlight {sev}.",
        "The streetlight pole {loc} is damaged and the light stays off.",
        "Streetlights {loc} remain switched off even after sunset {sev}.",
        "A streetlight {loc} is sparking and poses an electrical hazard.",
        "The area {loc} is completely dark at night since the streetlight stopped working {sev}.",
        "Broken streetlight wiring {loc} needs to be fixed urgently.",
        "Half the streetlights {loc} are non-functional {sev}, please repair them.",
        "The streetlight {loc} switches on only during the day, not at night.",
    ],
}


def build_dataset(samples_per_category=160):
    rows = []
    for category, templates in TEMPLATES.items():
        combos = set()
        attempts = 0
        while len(combos) < samples_per_category and attempts < samples_per_category * 20:
            attempts += 1
            template = random.choice(templates)
            loc = random.choice(LOCATIONS)
            sev = random.choice(SEVERITY_PHRASES)
            text = template.format(loc=loc, sev=sev)
            text = " ".join(text.split())
            combos.add(text)
        for text in combos:
            rows.append((text, category))
    random.shuffle(rows)
    return rows


def main():
    rows = build_dataset(samples_per_category=160)
    out_path = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "data", "complaints_text.csv"))
    with open(out_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["complaint_text", "category"])
        writer.writerows(rows)
    print(f"Wrote {len(rows)} rows to {out_path}")
    from collections import Counter
    print(Counter([r[1] for r in rows]))


if __name__ == "__main__":
    main()
