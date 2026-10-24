
import re
from pathlib import Path

KB_DIR = Path("data/knowledge_base")
RULE_RE = re.compile(r"^\s*(\d+)\.\s+(.*)")

for path in sorted(KB_DIR.glob("*.txt")):
    lines = path.read_text(encoding="utf-8").splitlines()

    numbers, lengths, orphans = [], [], []
    current_len = 0

    for i, line in enumerate(lines, start=1):
        m = RULE_RE.match(line)

        if m:
            if numbers:
                lengths.append(current_len)

            numbers.append(int(m.group(1)))
            current_len = len(m.group(2).split())

        elif line.strip():
            if numbers:
                current_len += len(line.split())  # Suite d'une règle
            else:
                orphans.append((i, line))  # Texte avant la règle 1

    if numbers:
        lengths.append(current_len)

    continu = numbers == list(range(1, len(numbers) + 1))

    print(f"== {path.name}")
    print(
        f"  Règles détectées : {len(numbers)} | "
        f"Numérotation continue : {continu}"
    )

    if lengths:
        print(
            f"  Mots par règle : min={min(lengths)} "
            f"max={max(lengths)} "
            f"moy={sum(lengths) / len(lengths):.1f}"
        )

    print(
        f"  Lignes avant la règle 1 : "
        f"{len(orphans)} {orphans[:2]}"
    )
