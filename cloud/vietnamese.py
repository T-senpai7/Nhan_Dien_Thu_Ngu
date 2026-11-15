"""Ghep dau phu (mu, trang, moc) va dau thanh vao tu dang danh van theo chinh ta tieng Viet."""
import unicodedata

TONE_MARKS = {
    "DAU_SAC": "\u0301",
    "DAU_HUYEN": "\u0300",
    "DAU_HOI": "\u0309",
    "DAU_NGA": "\u0303",
    "DAU_NANG": "\u0323",
}
SHAPE_MARKS = {
    "MU": {"A": "Â", "E": "Ê", "O": "Ô"},
    "DAU_TRANG": {"A": "Ă"},
    "MOC": {"U": "Ư", "O": "Ơ"},
}
VOWELS = set("AĂÂEÊIOÔƠUƯY")
MARKED_VOWELS = set("ĂÂÊÔƠƯ")


def strip_tone(ch: str) -> str:
    base = "".join(c for c in unicodedata.normalize("NFD", ch) if c not in TONE_MARKS.values())
    return unicodedata.normalize("NFC", base)


def tone_of(word: str):
    for ch in unicodedata.normalize("NFD", word):
        for name, mark in TONE_MARKS.items():
            if ch == mark:
                return name
    return None


def add_tone(ch: str, tone: str) -> str:
    return unicodedata.normalize("NFC", strip_tone(ch) + TONE_MARKS[tone]) if tone else ch


def tone_position(word: str):
    """Vi tri nguyen am mang dau thanh (quy tac kieu cu: hòa, thúy, toán, nghiêng, gì, quá)."""
    vowels = [i for i, c in enumerate(word) if c in VOWELS]
    if not vowels:
        return None
    end = start = vowels[-1]
    while start > 0 and word[start - 1] in VOWELS:
        start -= 1
    if start < end and start > 0 and word[start - 1:start + 1] in ("QU", "GI"):
        start += 1
    for i in range(end, start - 1, -1):
        if word[i] in MARKED_VOWELS:
            return i
    size = end - start + 1
    if size == 1:
        return start
    if size >= 3:
        return start + 1
    return end if end + 1 < len(word) else start


def apply_mark(word: str, mark: str):
    """Tra ve tu moi sau khi ghep dau, hoac None neu dau khong ap dung duoc cho tu nay."""
    if not word:
        return None
    tone = tone_of(word)
    plain = "".join(strip_tone(c) for c in word)

    if mark in TONE_MARKS:
        pos = tone_position(plain)
        if pos is None:
            return None
        return plain[:pos] + add_tone(plain[pos], mark) + plain[pos + 1:]

    table = SHAPE_MARKS.get(mark)
    if not table:
        return None
    chars = list(plain)
    for i in range(len(chars) - 1, -1, -1):
        if chars[i] in table:
            chars[i] = table[chars[i]]
            if mark == "MOC" and chars[i] == "Ơ" and i > 0 and chars[i - 1] == "U":
                chars[i - 1] = "Ư"
            break
    else:
        return None
    new = "".join(chars)
    if tone:
        pos = tone_position(new)
        new = new[:pos] + add_tone(new[pos], tone) + new[pos + 1:]
    return new
