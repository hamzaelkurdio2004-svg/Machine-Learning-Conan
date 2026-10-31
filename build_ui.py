# build_ui.py
# Builds conan_ui.html: one self-contained page (data embedded) from the model outputs.
#
#   python conanml_v2.py        -> conan_character_cards.csv + conan_future_probs.csv
#   python build_ui.py          -> conan_ui.html   (open it in any browser)
#
# Portraits and tones are NOT embedded. Drop your own files next to conan_ui.html:
#   assets/images/<key>.jpg   (or .png / .webp)      e.g. assets/images/heiji.jpg
#   assets/audio/<key>.mp3                           e.g. assets/audio/heiji.mp3
# Missing files fall back to a monogram placeholder / a "no tone yet" note.

import json
import sys

from pathlib import Path

import pandas as pd

HERE = Path(__file__).resolve().parent          # every file is looked up next to this script
TEMPLATE = HERE / "ui_template.html"
CARDS_CSV = HERE / "conan_character_cards.csv"
PROBS_CSV = HERE / "conan_future_probs.csv"
OUT_HTML = HERE / "conan_ui.html"


def check_files():
    needed = {
        TEMPLATE: "download ui_template.html and put it in this folder",
        CARDS_CSV: "run conanml_v2.py first (it creates this file)",
        PROBS_CSV: "run conanml_v2.py first (it creates this file)",
    }
    missing = [f"  - {p.name}: {how}" for p, how in needed.items() if not p.exists()]
    if missing:
        sys.exit(f"Missing files in {HERE}:\n" + "\n".join(missing))

# key | display name | group   (display-name words are searchable, so "bourbon" or "subaru" work too)
NAMES = """
conan|Conan Edogawa|Main cast
shinichi|Shinichi Kudo|Main cast
ran|Ran Mouri|Main cast
kogoro|Kogoro Mouri|Main cast
agasa|Dr. Hiroshi Agasa|Main cast
haibara|Ai Haibara|Detective Boys
ayumi|Ayumi Yoshida|Detective Boys
mitsuhiko|Mitsuhiko Tsuburaya|Detective Boys
genta|Genta Kojima|Detective Boys
megure|Inspector Juzo Megure|Police
takagi|Wataru Takagi|Police
sato|Miwako Sato|Police
shiratori|Ninzaburo Shiratori|Police
chiba|Kazunobu Chiba|Police
yumi|Yumi Miyamoto|Police
naeko|Naeko Miike|Police
matsumoto|Kiyonaga Matsumoto|Police
kuroda|Hyoue Kuroda|Police
yamato|Kansuke Yamato|Police
yui_uehara|Yui Uehara|Police
takaaki_morofushi|Takaaki Morofushi (Komei)|Police
heizo_hattori|Heizo Hattori|Police
ginshiro_toyama|Ginshiro Toyama|Police
otaki|Goro Otaki|Police
sango_yokomizo|Sango Yokomizo|Police
jugo_yokomizo|Jugo Yokomizo|Police
yamamura|Misao Yamamura|Police
chihaya_hagiwara|Chihaya Hagiwara|Police
jinpei_matsuda|Jinpei Matsuda|Police Academy
kenji_hagiwara|Kenji Hagiwara|Police Academy
wataru_date|Wataru Date|Police Academy
gin|Gin|Black Organization
vodka|Vodka|Black Organization
vermouth|Vermouth|Black Organization
bourbon_amuro|Tooru Amuro (Bourbon, Rei Furuya)|Black Organization
chianti|Chianti|Black Organization
korn|Korn|Black Organization
rum_wakita|Kanenori Wakita (Rum)|Black Organization
kir_rena|Rena Mizunashi (Kir)|Black Organization
pisco|Pisco|Black Organization
tequila|Tequila|Black Organization
calvados|Calvados|Black Organization
irish|Irish|Black Organization
curacao|Curacao|Black Organization
atsushi_miyano|Atsushi Miyano|Black Organization
elena_miyano|Elena Miyano|Black Organization
akemi_miyano|Akemi Miyano|Black Organization
suguru_itakura|Suguru Itakura|Black Organization
kiichiro_numabuchi|Kiichiro Numabuchi|Black Organization
renya_karasuma|Renya Karasuma|Black Organization
rikumichi_kusuda|Rikumichi Kusuda|Black Organization
akai_subaru|Shuichi Akai (Subaru Okiya)|FBI, CIA, PSB
jodie|Jodie Starling|FBI, CIA, PSB
james_black|James Black|FBI, CIA, PSB
camel|Andre Camel|FBI, CIA, PSB
eisuke_hondo|Eisuke Hondo|FBI, CIA, PSB
kazami|Yuya Kazami|FBI, CIA, PSB
scotch|Hiromitsu Morofushi (Scotch)|FBI, CIA, PSB
ethan_hondou|Ethan Hondou|FBI, CIA, PSB
heiji|Heiji Hattori|Kansai
kazuha|Kazuha Toyama|Kansai
shizuka_hattori|Shizuka Hattori|Kansai
iori_moga|Iori Muga|Kansai
momiji|Momiji Ooka|Kansai
kaito_kid|Kaito Kid (Kaito Kuroba)|Kaito Kid
nakamori|Inspector Ginzo Nakamori|Kaito Kid
jirokichi|Jirokichi Suzuki|Kaito Kid
konosuke_jii|Konosuke Jii|Kaito Kid
saguru_hakuba|Saguru Hakuba|Kaito Kid
toichi_kuroba|Toichi Kuroba|Kaito Kid
aoko_nakamori|Aoko Nakamori|Kaito Kid
yusaku_kudo|Yusaku Kudo|Family
yukiko_kudo|Yukiko Kudo|Family
eri_kisaki|Eri Kisaki|Family
midori_kuriyama|Midori Kuriyama|Family
masumi_sera|Masumi Sera|Family
mary_sera|Mary Sera|Family
shukichi_haneda|Shukichi Haneda|Family
koji_haneda|Koji Haneda|Family
sonoko|Sonoko Suzuki|Everyday cast
makoto_kyogoku|Makoto Kyogoku|Everyday cast
azusa_enomoto|Azusa Enomoto|Everyday cast
yoko_okino|Yoko Okino|Everyday cast
araide|Tomoaki Araide|Everyday cast
rumi_wakasa|Rumi Wakasa|Everyday cast
sumiko_kobayashi|Sumiko Kobayashi|Everyday cast
tsutomu_akai|Tsutomu Akai|Everyday cast
fusae_campbell|Fusae Campbell|Everyday cast
reiko_kujo|Reiko Kujo|Everyday cast
"""


def num(v, digits=0):
    return None if pd.isna(v) else (int(v) if digits == 0 else round(float(v), digits))


def build_payload():
    cards = pd.read_csv(CARDS_CSV).set_index("character")
    probs = pd.read_csv(PROBS_CSV)
    names = [line.split("|") for line in NAMES.strip().splitlines()]
    missing = set(cards.index) - {k for k, _, _ in names}
    assert not missing, f"add display names for: {missing}"

    chars = []
    for key, display, group in names:
        r = cards.loc[key]
        has_window = pd.notna(r.get("next_window_start"))
        chars.append({
            "k": key, "n": display, "g": group, "s": r["status"],
            "tot": int(r["total_appearances"]), "l50": int(r["appearances_last_50"]),
            "last": num(r["last_seen_episode"]),
            "w": [int(r["next_window_start"]), int(r["next_window_end"]),
                num(r["next_window_prob_%"], 1)] if has_window else None,
            "p100": num(r.get("prob_at_least_once_in_100_%"), 1),
            "e100": num(r.get("expected_appearances_in_100"), 1),
            "pr": [int(round(p * 100)) for p in probs[key]] if key in probs.columns else [],
        })
    return {
        "start": int(probs["episode_no"].iloc[0]),
        "n": len(probs),
        "pc": [int(round(p * 100)) for p in probs["p_canon"]],
        "chars": chars,
    }


def main():
    check_files()
    fragment_out = sys.argv[sys.argv.index("--fragment") + 1] if "--fragment" in sys.argv else None
    html = TEMPLATE.read_text(encoding="utf-8")
    data = json.dumps(build_payload(), separators=(",", ":")).replace("</", "<\\/")
    html = html.replace("/*__DATA__*/null", data)

    if fragment_out:  # fragment only (for hosts that add their own <html>/<head>)
        open(fragment_out, "w", encoding="utf-8").write(html)
        return
    page = ('<!doctype html><html lang="en"><head><meta charset="utf-8">'
            '<meta name="viewport" content="width=device-width, initial-scale=1"></head><body>'
            + html + "</body></html>")
    OUT_HTML.write_text(page, encoding="utf-8")
    print(f"{OUT_HTML} written ({len(page) / 1024:.0f} KB). Put assets/images and assets/audio next to it.")


if __name__ == "__main__":
    main()
