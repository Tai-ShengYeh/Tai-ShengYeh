#!/usr/bin/env python3
"""Download an IRUG spectrum, save it as CSV, and plot it.

IRUG (the Infrared and Raman Users Group, http://www.irug.org) publishes
reference spectra as JCAMP-DX files. Each spectrum has a detail page such as

    http://www.irug.org/jcamp-details?id=3537

This script:

  1. Fetches the JCAMP-DX data for a spectrum (by id, by detail-page URL, by
     direct JCAMP-DX URL, or from a local .jdx/.dx/.txt file).
  2. Parses the JCAMP-DX file (supports plain AFFN, PAC, and the compressed
     ASDF forms SQZ/DIF/DUP, plus (XY..XY) peak/xy tables).
  3. Writes the (x, y) data to a CSV file.
  4. Plots the spectrum with matplotlib (wavenumber axis reversed for IR).

Examples
--------
    python irug_spectrum.py                          # default id = 3537
    python irug_spectrum.py --id 3537
    python irug_spectrum.py --url "http://www.irug.org/jcamp-details?id=3537"
    python irug_spectrum.py --jcamp spectrum.jdx     # parse a local file
    python irug_spectrum.py --id 3537 --csv out.csv --plot out.png --no-show

Requirements
------------
    pip install matplotlib        # plotting
    pip install requests          # optional; falls back to urllib

Note
----
Some networks/firewalls block irug.org. If the download fails, open the
detail page in a browser, use its "Download" button to save the JCAMP-DX
file, then run this script with --jcamp <file>.
"""

from __future__ import annotations

import argparse
import csv
import os
import re
import sys
import urllib.request

DEFAULT_ID = 3537
USER_AGENT = (
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
    "(KHTML, like Gecko) Chrome/124.0 Safari/537.36"
)


# --------------------------------------------------------------------------- #
# Fetching
# --------------------------------------------------------------------------- #
def _http_get(url: str, accept: str = "*/*") -> bytes:
    """GET a URL with a browser-like User-Agent. Uses requests if available."""
    try:
        import requests  # type: ignore

        resp = requests.get(
            url, headers={"User-Agent": USER_AGENT, "Accept": accept}, timeout=60
        )
        resp.raise_for_status()
        return resp.content
    except ImportError:
        req = urllib.request.Request(
            url, headers={"User-Agent": USER_AGENT, "Accept": accept}
        )
        with urllib.request.urlopen(req, timeout=60) as r:  # noqa: S310
            return r.read()


def _looks_like_jcamp(text: str) -> bool:
    return "##TITLE=" in text or "##XYDATA=" in text or "##XYPOINTS=" in text


def _extract_inline_jcamp(html: str) -> str | None:
    """Return an inline JCAMP-DX block embedded in a detail page, if present."""
    m = re.search(r"(##TITLE=.*?##END=\s*)", html, re.DOTALL | re.IGNORECASE)
    return m.group(1) if m else None


def _find_jcamp_link(html: str, base_url: str) -> str | None:
    """Find a link to a JCAMP-DX file (.jdx/.dx/.jcamp/.txt) in page HTML."""
    candidates = re.findall(
        r"""(?:href|src|data-[\w-]*)\s*=\s*['"]([^'"]+\.(?:jdx|dx|jcamp|jcm|txt))['"]""",
        html,
        re.IGNORECASE,
    )
    if not candidates:
        return None
    return urllib.request.urljoin(base_url, candidates[0])


def fetch_jcamp(*, jcamp_id=None, url=None, jcamp_file=None) -> str:
    """Return JCAMP-DX text from a local file, a direct URL, or an IRUG page."""
    if jcamp_file:
        with open(jcamp_file, "r", encoding="utf-8", errors="replace") as fh:
            return fh.read()

    if url is None:
        if jcamp_id is None:
            jcamp_id = DEFAULT_ID
        url = f"http://www.irug.org/jcamp-details?id={jcamp_id}"

    raw = _http_get(url, accept="chemical/x-jcamp-dx, text/plain, text/html")
    text = raw.decode("utf-8", errors="replace")

    # The URL pointed straight at a JCAMP-DX file.
    if _looks_like_jcamp(text) and not text.lstrip().lower().startswith("<!doctype"):
        return text

    # It's an HTML detail page: try inline data, then a download link.
    inline = _extract_inline_jcamp(text)
    if inline:
        return inline

    link = _find_jcamp_link(text, url)
    if link:
        return _http_get(link, accept="chemical/x-jcamp-dx, text/plain").decode(
            "utf-8", errors="replace"
        )

    raise RuntimeError(
        "Could not locate JCAMP-DX data on the page.\n"
        f"  URL: {url}\n"
        "Open the page in a browser, download the spectrum's JCAMP-DX file "
        "via its Download button, then re-run with:  --jcamp <file>"
    )


# --------------------------------------------------------------------------- #
# JCAMP-DX parsing
# --------------------------------------------------------------------------- #
# ASDF (ASCII Squeezed Difference Form) pseudo-digit tables.
_SQZ = {  # absolute value, leading digit + sign
    "@": "+0", "A": "+1", "B": "+2", "C": "+3", "D": "+4",
    "E": "+5", "F": "+6", "G": "+7", "H": "+8", "I": "+9",
    "a": "-1", "b": "-2", "c": "-3", "d": "-4",
    "e": "-5", "f": "-6", "g": "-7", "h": "-8", "i": "-9",
}
_DIF = {  # difference from previous ordinate
    "%": "+0", "J": "+1", "K": "+2", "L": "+3", "M": "+4",
    "N": "+5", "O": "+6", "P": "+7", "Q": "+8", "R": "+9",
    "j": "-1", "k": "-2", "l": "-3", "m": "-4",
    "n": "-5", "o": "-6", "p": "-7", "q": "-8", "r": "-9",
}
_DUP = {  # duplicate-count of the previous ordinate
    "S": "1", "T": "2", "U": "3", "V": "4",
    "W": "5", "X": "6", "Y": "7", "Z": "8", "s": "9",
}


def _parse_ldrs(text: str) -> dict:
    """Parse JCAMP labelled-data-records (header) into a dict of strings."""
    ldrs: dict[str, str] = {}
    key = None
    for raw_line in text.splitlines():
        line = raw_line.split("$$", 1)[0].rstrip()  # strip comments
        if line.startswith("##"):
            label, _, value = line[2:].partition("=")
            key = label.strip().upper().replace(" ", "").replace("-", "").replace("_", "")
            ldrs[key] = value.strip()
        elif key is not None and line:
            ldrs[key] += "\n" + line
    return ldrs


def _tokenize_asdf(data: str):
    """Yield (mode, number_string) tokens from one ASDF/AFFN data string.

    mode is one of: 'NUM' (plain/PAC/SQZ absolute), 'DIF', 'DUP'.
    """
    tokens = []
    cur = ""
    mode = None

    def flush():
        nonlocal cur, mode
        if cur not in ("", "+", "-"):
            tokens.append((mode, cur))
        cur, mode = "", None

    for ch in data:
        if ch in _SQZ:
            flush()
            cur, mode = _SQZ[ch], "NUM"
        elif ch in _DIF:
            flush()
            cur, mode = _DIF[ch], "DIF"
        elif ch in _DUP:
            flush()
            cur, mode = _DUP[ch], "DUP"
        elif ch in "+-":
            flush()
            cur, mode = ("-" if ch == "-" else ""), "NUM"
        elif ch.isdigit() or ch == ".":
            if mode is None:
                mode = "NUM"
            cur += ch
        elif ch in "eE" and cur and (cur[-1].isdigit() or cur[-1] == "."):
            cur += ch  # exponent of a plain number
        else:  # whitespace, commas, anything else -> delimiter
            flush()
    flush()
    return tokens


def _parse_xydata(data_block: str, xfactor: float, yfactor: float):
    """Decode an ##XYDATA=(X++(Y..Y)) block into (x_list, y_list)."""
    uses_dif = any(c in _DIF for c in data_block)
    xs: list[float] = []
    ys: list[float] = []

    for raw_line in data_block.splitlines():
        line = raw_line.split("$$", 1)[0].strip()
        if not line:
            continue
        tokens = _tokenize_asdf(line)
        if not tokens:
            continue

        # First token is the line's X (AFFN, absolute).
        line_x = float(tokens[0][1])
        y_tokens = tokens[1:]

        last_y = ys[-1] if ys else None
        for idx, (mode, num) in enumerate(y_tokens):
            if mode == "DUP":
                if last_y is None:
                    continue
                ys.extend([last_y] * (int(num) - 1))  # value already emitted once
                continue
            if mode == "DIF":
                last_y = last_y + float(num)
            else:  # NUM (absolute)
                val = float(num)
                # DIF files repeat the previous line's last ordinate as the
                # first ordinate of the next line (a redundancy check) -> skip.
                if (
                    idx == 0
                    and uses_dif
                    and ys
                    and last_y is not None
                    and abs(val - last_y) < 1e-6
                ):
                    continue
                last_y = val
            ys.append(last_y)

        # Record the starting X of this line; expand X uniformly afterwards.
        xs.append(line_x * xfactor)

    return xs, ys


def _parse_xypoints(data_block: str, xfactor: float, yfactor: float):
    """Decode an ##XYPOINTS / ##PEAKTABLE (XY..XY) block into (x, y)."""
    nums = re.findall(r"[-+]?\d*\.?\d+(?:[eE][-+]?\d+)?", data_block)
    vals = [float(n) for n in nums]
    xs = [vals[i] * xfactor for i in range(0, len(vals) - 1, 2)]
    ys = [vals[i] * yfactor for i in range(1, len(vals), 2)]
    return xs, ys


def parse_jcamp(text: str):
    """Parse JCAMP-DX text into (x, y, meta).

    Returns
    -------
    x, y : lists of float
    meta : dict with keys like title, xunits, yunits.
    """
    # Prefer the well-tested `jcamp` package when it's installed.
    try:
        from jcamp import jcamp_read  # type: ignore
        import io as _io

        d = jcamp_read(_io.StringIO(text))
        x = list(d.get("x", []))
        y = list(d.get("y", []))
        if x and y:
            meta = {
                "title": d.get("title", "IRUG spectrum"),
                "xunits": d.get("xunits", ""),
                "yunits": d.get("yunits", ""),
            }
            return x, y, meta
    except Exception:
        pass  # fall back to the built-in parser

    ldrs = _parse_ldrs(text)
    xfactor = float(ldrs.get("XFACTOR", "1") or "1")
    yfactor = float(ldrs.get("YFACTOR", "1") or "1")
    meta = {
        "title": ldrs.get("TITLE", "IRUG spectrum"),
        "xunits": ldrs.get("XUNITS", ""),
        "yunits": ldrs.get("YUNITS", ""),
    }

    if "XYDATA" in ldrs:
        block = ldrs["XYDATA"]
        block = block.split("\n", 1)[1] if "\n" in block else ""  # drop the (X++(Y..Y)) header
        xs, raw_ys = _parse_xydata(block, xfactor, yfactor)
        ys = [v * yfactor for v in raw_ys]

        # Reconstruct a uniform X grid from FIRSTX/LASTX/NPOINTS when available
        # (the per-line X values only mark line starts).
        n = len(ys)
        firstx = ldrs.get("FIRSTX")
        lastx = ldrs.get("LASTX")
        if firstx is not None and lastx is not None and n > 1:
            fx, lx = float(firstx), float(lastx)
            step = (lx - fx) / (n - 1)
            x = [fx + i * step for i in range(n)]
        elif xs and n > 1:
            x = [xs[0] + (xs[-1] - xs[0]) * i / (n - 1) for i in range(n)]
        else:
            x = xs
        return x, ys, meta

    for key in ("XYPOINTS", "PEAKTABLE", "XYDATA"):
        if key in ldrs:
            block = ldrs[key]
            block = block.split("\n", 1)[1] if "\n" in block else block
            x, y = _parse_xypoints(block, xfactor, yfactor)
            return x, y, meta

    raise RuntimeError("No ##XYDATA/##XYPOINTS data record found in the JCAMP file.")


# --------------------------------------------------------------------------- #
# Output
# --------------------------------------------------------------------------- #
def save_csv(x, y, meta, path: str) -> None:
    xlabel = meta.get("xunits") or "x"
    ylabel = meta.get("yunits") or "y"
    with open(path, "w", newline="", encoding="utf-8") as fh:
        w = csv.writer(fh)
        w.writerow([xlabel, ylabel])
        w.writerows(zip(x, y))
    print(f"Wrote {len(x)} points to {path}")


def plot_spectrum(x, y, meta, png_path=None, show=True) -> None:
    try:
        import matplotlib

        if not show:
            matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        print("matplotlib not installed; skipping plot (pip install matplotlib).")
        return

    xlabel = meta.get("xunits") or "x"
    ylabel = meta.get("yunits") or "y"

    fig, ax = plt.subplots(figsize=(10, 5))
    ax.plot(x, y, lw=0.8)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.set_title(meta.get("title", "IRUG spectrum"))
    ax.grid(True, alpha=0.3)

    # IR convention: wavenumber decreases left-to-right.
    if re.search(r"(wavenumber|1/cm|cm-1|cm\^-1)", xlabel, re.IGNORECASE):
        ax.invert_xaxis()

    fig.tight_layout()
    if png_path:
        fig.savefig(png_path, dpi=150)
        print(f"Saved plot to {png_path}")
    if show:
        plt.show()


# --------------------------------------------------------------------------- #
# CLI
# --------------------------------------------------------------------------- #
def main(argv=None) -> int:
    p = argparse.ArgumentParser(description="Download an IRUG spectrum to CSV and plot it.")
    src = p.add_mutually_exclusive_group()
    src.add_argument("--id", type=int, help=f"IRUG spectrum id (default {DEFAULT_ID}).")
    src.add_argument("--url", help="IRUG detail-page URL or direct JCAMP-DX URL.")
    src.add_argument("--jcamp", help="Path to a local JCAMP-DX (.jdx/.dx/.txt) file.")
    p.add_argument("--csv", help="Output CSV path (default spectrum_<id>.csv).")
    p.add_argument("--plot", help="Output PNG path for the plot.")
    p.add_argument("--no-show", action="store_true", help="Do not open an interactive window.")
    args = p.parse_args(argv)

    jcamp_id = args.id if args.id is not None else (None if (args.url or args.jcamp) else DEFAULT_ID)

    try:
        text = fetch_jcamp(jcamp_id=jcamp_id, url=args.url, jcamp_file=args.jcamp)
    except Exception as exc:  # noqa: BLE001
        print(f"Error fetching JCAMP-DX: {exc}", file=sys.stderr)
        return 1

    try:
        x, y, meta = parse_jcamp(text)
    except Exception as exc:  # noqa: BLE001
        print(f"Error parsing JCAMP-DX: {exc}", file=sys.stderr)
        return 1

    if not x or not y:
        print("No data points parsed from the JCAMP file.", file=sys.stderr)
        return 1

    tag = args.id if args.id is not None else (jcamp_id or "irug")
    csv_path = args.csv or f"spectrum_{tag}.csv"
    save_csv(x, y, meta, csv_path)
    plot_spectrum(x, y, meta, png_path=args.plot, show=not args.no_show)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
