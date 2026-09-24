"""Surface cues (docs/tracks.md §9): the words of a record and its context must not predict
the answer. A bag-of-words Bernoulli naive Bayes over state + instructions, trained on
public decisions cases, must not beat the majority class on determined yes/no items."""

import math
import re
from collections import Counter

from opentype_challenge import generator as g

WORD = re.compile(r"[a-z0-9_]+")


def items(seed: int, per_level: int) -> list[tuple[frozenset[str], str]]:
    out = []
    for level in (1, 2, 3, 4):
        for case in g.generate(None, level, per_level, seed):
            text = f"{case.body['state']}\n{case.body['instructions']}".lower()
            words = frozenset(WORD.findall(text))
            for gold in case.gold.values():
                if gold.kind == "noul" and gold.determined:
                    out.append((words, gold.options[gold.probs.index(1.0)]))
    return out


class NaiveBayes:
    def __init__(self, data: list[tuple[frozenset[str], str]]) -> None:
        self.labels = Counter(label for _, label in data)
        seen: dict[str, Counter[str]] = {label: Counter() for label in self.labels}
        for words, label in data:
            seen[label].update(words)
        vocab = set().union(*seen.values())
        self.present: dict[str, dict[str, float]] = {}
        self.absent: dict[str, float] = {}
        for label, n in self.labels.items():
            p = {w: (seen[label][w] + 1) / (n + 2) for w in vocab}
            self.present[label] = {w: math.log(q) - math.log(1 - q) for w, q in p.items()}
            self.absent[label] = math.log(n) + sum(math.log(1 - q) for q in p.values())

    def predict(self, words: frozenset[str]) -> str:
        def score(label: str) -> float:
            table = self.present[label]
            return self.absent[label] + sum(table.get(w, 0.0) for w in words)

        return max(sorted(self.labels), key=score)


def test_bag_of_words_does_not_beat_the_majority_class():
    train = items(seed=0, per_level=750)  # 3000 cases
    test = items(seed=1, per_level=250)  # 1000 others
    assert len(train) > 3000 and len(test) > 1000
    model = NaiveBayes(train)
    majority = model.labels.most_common(1)[0][0]
    baseline = sum(label == majority for _, label in test) / len(test)
    accuracy = sum(model.predict(words) == label for words, label in test) / len(test)
    assert abs(accuracy - baseline) <= 0.03, (accuracy, baseline)
