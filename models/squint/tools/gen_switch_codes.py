"""Writes switch_codes.json: a few QR module grids made with Segno for the payload-switch sequences.

Pairs share version, error correction level and mask, so the two grids differ in few modules (one changed character).
Run: python gen_switch_codes.py   (needs `pip install segno`)
"""
import json
import segno

SPECS = [
    # id, payload text, version, ecc, mode, mask
    ("sw_a3", "ORDER 4417 CONFIRMED", 3, "M", "byte", 2),
    ("sw_b3", "ORDER 4471 CONFIRMED", 3, "M", "byte", 2),
    ("sw_n5a", "202610301234567890123456789012", 5, "Q", "numeric", 3),
    ("sw_n5b", "202610301234567890123456789013", 5, "Q", "numeric", 3),
    ("sw_c2", "https://seal.example/a", 2, "M", "byte", 1),
    ("sw_c4", "https://seal.example/a/b/c/d?x=1", 4, "M", "byte", 1),
    ("sw_c6", "WIFI:T:WPA;S:SealGuest;P:correct horse battery staple;;", 6, "L", "byte", 5),
]
out = []
for ident, text, version, ecc, mode, mask in SPECS:
    q = segno.make(text, version=version, error=ecc, mode=mode, mask=mask, micro=False, boost_error=False)
    assert q.version == version and q.mask == mask, (ident, q.version, q.mask)
    grid = "".join("".join("1" if b else "0" for b in row) for row in q.matrix)
    n = len(q.matrix)
    assert n == 17 + 4 * version and len(grid) == n * n
    out.append({"id": ident, "version": version, "ecc": ecc, "mask": mask, "mode": mode,
                "payloadHex": text.encode("utf-8").hex(), "grid": grid})
with open("switch_codes.json", "w") as f:
    json.dump(out, f, indent=1)
print("wrote", len(out), "codes")
