"""Hand-built features shared by training and serving."""
import re

NUM_WORDS = {
    "zero": 0, "one": 1, "a": 1, "an": 1, "two": 2, "three": 3, "four": 4, "five": 5,
    "six": 6, "seven": 7, "eight": 8, "nine": 9, "ten": 10, "eleven": 11, "twelve": 12,
    "fourteen": 14, "fifteen": 15, "twenty": 20, "thirty": 30, "forty-eight": 48,
    "forty eight": 48, "sixty": 60, "ninety": 90, "half": 0.5, "a few": 3,
}
UNIT_DAYS = {
    "minute": 1 / 1440, "hour": 1 / 24, "day": 1, "business day": 1, "week": 7,
    "fortnight": 14, "month": 30, "year": 365,
}
_num = r"(\d+(?:\.\d+)?|" + "|".join(sorted(map(re.escape, NUM_WORDS), key=len, reverse=True)) + r")"
_unit = r"(business days?|minutes?|hours?|days?|weeks?|fortnights?|months?|years?)"
QTY_RE = re.compile(r"\b" + _num + r"[\s-]+(?:\w+\s+)?" + _unit + r"\b", re.I)
HALF_HOUR = re.compile(r"\bhalf an? hour\b", re.I)


def quantities(text):
    """Durations normalised to days, rounded so 'two weeks' == '14 days'."""
    out = set()
    t = text.lower()
    if HALF_HOUR.search(t):
        out.add(round(30 / 1440, 4))
    for n, u in QTY_RE.findall(t):
        n = NUM_WORDS.get(n, None) if not n[0].isdigit() else float(n)
        if n is None:
            continue
        u = u.rstrip("s") if u != "business days" else "business day"
        out.add(round(float(n) * UNIT_DAYS.get(u, 1), 4))
    return out


def tokens(text):
    return re.findall(r"[a-z0-9]+", text.lower())
