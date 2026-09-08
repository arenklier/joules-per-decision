"""Fetch hourly Turkish grid generation mix and derive carbon intensity.

Run this yourself -- it needs an EPIAS transparency account and this script
never sees those credentials except from your own environment:

    export EPIAS_USER='...'          # or set EPIAS_USER=... on Windows
    export EPIAS_PASS='...'
    python epias_fetch.py 2026-08-01 2026-09-01

It writes carbon_intensity_hourly.csv, which is the only file that needs to
come back for the analysis. Register free at https://kayit.epias.com.tr .

Why hourly and not an annual average: the point of the carbon layer is that an
energy-optimal schedule and a carbon-optimal one are not the same schedule.
Campus and operations workloads shift naturally toward the night, and the
Turkish grid's mix at 03:00 is not its mix at 13:00. An annual mean erases
exactly the effect worth reporting.

Emission factors are IPCC AR5 (WG3, Annex III) lifecycle medians, gCO2eq/kWh.
Swap them if you prefer a national inventory factor -- the column names below
are the only thing the analysis depends on.
"""
import csv
import json
import os
import sys
import urllib.parse
import urllib.request

CAS = "https://giris.epias.com.tr/cas/v1/tickets"
GEN = ("https://seffaflik.epias.com.tr/electricity-service/v1/generation/"
       "data/realtime-generation")

# IPCC AR5 WG3 Annex III lifecycle medians, gCO2eq/kWh.
FACTORS = {
    "importCoal": 820, "asphaltiteCoal": 820, "lignite": 820, "blackCoal": 820,
    "naturalGas": 490, "fueloil": 650, "gasOil": 650, "lng": 490,
    "biomass": 230, "naphta": 650,
    "river": 24, "dammedHydro": 24, "geothermal": 38,
    "wind": 11, "sun": 48, "nuclear": 12, "wasteheat": 260, "other": 490,
}


def get_tgt(user, password):
    body = urllib.parse.urlencode({"username": user, "password": password}).encode()
    req = urllib.request.Request(
        CAS, data=body,
        headers={"Content-Type": "application/x-www-form-urlencoded",
                 "Accept": "text/plain"})
    with urllib.request.urlopen(req, timeout=30) as r:
        # the ticket comes back either in the body or in the Location header
        text = r.read().decode(errors="replace").strip()
        loc = r.headers.get("Location", "")
    for candidate in (text, loc):
        if "TGT-" in candidate:
            return candidate[candidate.index("TGT-"):].strip().strip('"')
    raise SystemExit("could not parse a TGT from the CAS response")


def fetch(tgt, start, end):
    body = json.dumps({"startDate": f"{start}T00:00:00+03:00",
                       "endDate": f"{end}T00:00:00+03:00"}).encode()
    req = urllib.request.Request(
        GEN, data=body,
        headers={"Content-Type": "application/json", "Accept": "application/json",
                 "TGT": tgt})
    with urllib.request.urlopen(req, timeout=120) as r:
        return json.load(r).get("items", [])


def main():
    if len(sys.argv) != 3:
        raise SystemExit("usage: epias_fetch.py YYYY-MM-DD YYYY-MM-DD")
    user, password = os.environ.get("EPIAS_USER"), os.environ.get("EPIAS_PASS")
    if not user or not password:
        raise SystemExit("set EPIAS_USER and EPIAS_PASS in the environment first")

    rows = fetch(get_tgt(user, password), sys.argv[1], sys.argv[2])
    print(f"{len(rows)} hourly records")
    if not rows:
        return

    known = [k for k in FACTORS if k in rows[0]]
    missing = [k for k in rows[0] if k not in FACTORS
               and k not in ("date", "hour", "time", "total")]
    if missing:
        print(f"note: no emission factor for {missing} -- excluded from the mix")

    out = "carbon_intensity_hourly.csv"
    with open(out, "w", newline="", encoding="utf-8") as fh:
        w = csv.writer(fh)
        w.writerow(["timestamp", "total_mwh", "gco2_per_kwh"] + known)
        for r in rows:
            mix = {k: float(r.get(k) or 0) for k in known}
            total = sum(mix.values())
            if total <= 0:
                continue
            gco2 = sum(v * FACTORS[k] for k, v in mix.items()) / total
            w.writerow([r.get("date") or r.get("time"), round(total, 2),
                        round(gco2, 2)] + [round(mix[k], 2) for k in known])

    with open(out, encoding="utf-8") as fh:
        vals = [float(x["gco2_per_kwh"]) for x in csv.DictReader(fh)]
    if vals:
        print(f"wrote {out}")
        print(f"carbon intensity: mean {sum(vals)/len(vals):.0f}, "
              f"min {min(vals):.0f}, max {max(vals):.0f} gCO2eq/kWh")
        print(f"peak-to-trough spread: {max(vals)/min(vals):.2f}x "
              f"-- this ratio is the finding; a flat grid would make it 1.0")


if __name__ == "__main__":
    main()
