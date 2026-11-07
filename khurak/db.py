"""Product database and brand / ingredient indexes."""
import gzip
import os
import re
from functools import lru_cache

from .normalize import SYNONYMS, canonical

DATA = os.path.join(os.path.dirname(__file__), "data")


class Product:
    __slots__ = ("id", "name", "brand", "variant", "manufacturer", "price", "discontinued", "pack",
                 "ingredients", "strengths", "mg", "complete", "missing_hint", "source")

    def __init__(self, row):
        (self.id, self.name, self.brand, self.variant, self.manufacturer, price, disc, self.pack, ings, strs, mg,
         comp, hint, self.source) = row
        self.price = float(price) if price else None
        self.discontinued = disc == "1"
        self.ingredients = ings.split("|")
        self.strengths = strs.split("|")
        self.mg = [float(x) if x else None for x in mg.split("|")]
        self.complete = comp == "1"
        self.missing_hint = hint == "1"

    def composition(self):
        return " + ".join(f"{i} {s}".strip() for i, s in zip(self.ingredients, self.strengths))

    def as_dict(self):
        return {"id": self.id, "name": self.name, "manufacturer": self.manufacturer, "price": self.price,
                "discontinued": self.discontinued, "ingredients": self.ingredients, "strengths": self.strengths,
                "complete": self.complete, "missing_hint": self.missing_hint, "source": self.source}


def base_key(k):
    return k.split("-")[0]


class DB:
    def __init__(self, path=None):
        path = path or os.path.join(DATA, "products.tsv.gz")
        self.products = []
        with gzip.open(path, "rt", encoding="utf-8") as f:
            next(f)
            for line in f:
                self.products.append(Product(line.rstrip("\n").split("\t")))
        self.ingredients = sorted({i for p in self.products for i in p.ingredients})
        self.generic_names = set(self.ingredients) | set(SYNONYMS)
        # single-word forms of ingredient names route to the generic path, not the brand path
        generic_first = {g.split()[0] for g in self.generic_names}
        self.brands = {}
        for i, p in enumerate(self.products):
            for k in {p.brand, base_key(p.brand)}:
                if len(k) >= 3 and k not in generic_first and not k.isdigit():
                    self.brands.setdefault(k, []).append(i)
        self._deletes = None

    # ---- fuzzy lookup (symmetric delete, edit distance 1) for keys of 5+ chars
    def _build_deletes(self):
        d = {}
        for k in self.brands:
            if len(k) >= 5:
                for i in range(len(k)):
                    d.setdefault(k[:i] + k[i + 1:], set()).add(k)
                d.setdefault(k, set()).add(k)
        self._deletes = d

    def fuzzy(self, word):
        if self._deletes is None:
            self._build_deletes()
        if len(word) < 5:
            return set()
        out = set(self._deletes.get(word, ()))
        for i in range(len(word)):
            out |= self._deletes.get(word[:i] + word[i + 1:], set())
        return {k for k in out if abs(len(k) - len(word)) <= 1}

    def family(self, key):
        return [self.products[i] for i in self.brands.get(key, [])]


@lru_cache(maxsize=1)
def default_db():
    return DB()
