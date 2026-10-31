import pandas as pd
import requests
from bs4 import BeautifulSoup
import urllib.parse
import time


def extract_cast_links(soup):
    # cleaning the web for the side bars and the trashs 
    for trash in soup.find_all(
        ["div", "nav"],
        class_=[
            "toc",
            "mw-toc",
            "navbox",
            "vertical-navbox",
            "toclimit-2",
        ],
    ):
        trash.decompose()

    cast_h2 = None # we gave him a startup value = 0 
    # search for every header if it contains cast we add it 
    for h2 in soup.find_all("h2"):
        if "cast" in h2.get_text().strip().lower():
            cast_h2 = h2
            break

    if not cast_h2: # if there is no cast we return an empty List 
        return []

    start_node = cast_h2 # we start from the box that contains cast 
    while start_node.parent and start_node.parent.name not in ["body", "html"]: # resume doing the task as long as we are still in cast
        # search if it contains a unique card (id) if not we got an empty text
        p_id = start_node.parent.get("id", "") 
        # search if it contains a unique stikcer (class) if not we got an empty text
        p_class = start_node.parent.get("class", []) 
        # turn the stickers into connected text so we can easily find the words in it 
        p_class_str = (
            " ".join(p_class) if isinstance(p_class, list) else str(p_class)
        )
        # it get us to the main box in the article it means we need to stop over here
        if p_id == "mw-content-text" or "mw-parser-output" in p_class_str:
            break
        # if we havent get to the main box move to the other box 
        start_node = start_node.parent

    links = []
    # move box to box 
    for sibling in start_node.find_next_siblings():
        # if we got to a new header stop 
        if sibling.name == "h2" or sibling.find("h2"):
            break

        s_class = sibling.get("class", [])
        s_class_str = (
            " ".join(s_class) if isinstance(s_class, list) else str(s_class)
        )
        if "mw-heading2" in s_class_str:
            break

        # search into the boces if it has a link and go from one to another
        for a in sibling.find_all("a", href=True):
            raw_href = a["href"] # it get the link row like /wiki/Takaaki_Morofushi_%28Koumei%29
            unquoted_href = urllib.parse.unquote(raw_href) # it make the text clean like /wiki/Takaaki_Morofushi_(Koumei)
            links.append(raw_href)
            links.append(unquoted_href)

    return list(set(links))


def scrape_full_dataset():
    print("⏳ Reading 'conan_episodes_clean.csv'...")
    try:
        df = pd.read_csv("conan_episodes_clean.csv")
    except FileNotFoundError:
        print("❌ File 'conan_episodes_clean.csv' not found!")
        return

    # keep the request safe from getting blocked
    session = requests.Session() 
    session.headers.update({
        "User-Agent": (
            "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
            "AppleWebKit/537.36 (KHTML, like Gecko) "
            "Chrome/122.0.0.0 Safari/537.36"
        )
    })

    # القاموس الشامل والمحدث بدقة 100% لجميع الـ Slugs
    character_slugs = {
        # 1️⃣ Main Cast & Detective Boys
        "conan": ["/wiki/Conan_Edogawa"],
        "shinichi": ["/wiki/Shinichi_Kudo"],
        "ran": ["/wiki/Ran_Mouri"],
        "kogoro": ["/wiki/Kogoro_Mouri"],
        "agasa": ["/wiki/Hiroshi_Agasa"],
        "haibara": ["/wiki/Ai_Haibara"],
        "ayumi": ["/wiki/Ayumi_Yoshida"],
        "mitsuhiko": ["/wiki/Mitsuhiko_Tsuburaya"],
        "genta": ["/wiki/Genta_Kojima"],
        # 2️⃣ Tokyo & Prefectures Police
        "megure": ["/wiki/Juzo_Megure"],
        "takagi": ["/wiki/Wataru_Takagi"],
        "sato": ["/wiki/Miwako_Sato"],
        "shiratori": ["/wiki/Ninzaburo_Shiratori"],
        "chiba": ["/wiki/Kazunobu_Chiba"],
        "yumi": ["/wiki/Yumi_Miyamoto"],
        "naeko": ["/wiki/Naeko_Miike"],
        "matsumoto": ["/wiki/Kiyonaga_Matsumoto"],
        "kuroda": ["/wiki/Hyoue_Kuroda"],
        "yamato": ["/wiki/Kansuke_Yamato"],
        "yui_uehara": ["/wiki/Yui_Uehara"],
        "takaaki_morofushi": [
            "/wiki/Taka%27aki_Morofushi_(Komei)",
            "/wiki/Taka'aki_Morofushi_(Komei)",
            "/wiki/Taka%27aki_Morofushi_%28Komei%29",
            "/wiki/Taka'aki_Morofushi_%28Komei%29",
            "/wiki/Taka%27aki_Morofushi_(Koumei)",
            "/wiki/Taka'aki_Morofushi_(Koumei)",
            "/wiki/Taka'aki_Morofushi",
            "/wiki/Taka%27aki_Morofushi",
            "/wiki/Takaaki_Morofushi",
            "/wiki/Komei",
            "/wiki/Koumei",
        ],
        "heizo_hattori": ["/wiki/Heizo_Hattori"],
        "ginshiro_toyama": ["/wiki/Ginshiro_Toyama"],
        "otaki": ["/wiki/Goro_Otaki"],
        "sango_yokomizo": ["/wiki/Sango_Yokomizo"],
        "jugo_yokomizo": ["/wiki/Jugo_Yokomizo"],
        "yamamura": ["/wiki/Misao_Yamamura"],
        "chihaya_hagiwara": ["/wiki/Chihaya_Hagiwara"],
        # 3️⃣ Police Academy & Past Cases
        "jinpei_matsuda": ["/wiki/Jinpei_Matsuda"],
        "kenji_hagiwara": ["/wiki/Kenji_Hagiwara"],
        "wataru_date": ["/wiki/Wataru_Date"],
        # 4️⃣ Black Organization & Related
        "gin": ["/wiki/Gin"],
        "vodka": ["/wiki/Vodka"],
        "vermouth": ["/wiki/Vermouth"],
        "bourbon_amuro": ["/wiki/Tooru_Amuro", "/wiki/Rei_Furuya"],
        "chianti": ["/wiki/Chianti"],
        "korn": ["/wiki/Korn"],
        "rum_wakita": ["/wiki/Kanenori_Wakita", "/wiki/Rum"],
        "kir_rena": [
            "/wiki/Rena_Mizunashi",
            "/wiki/Kir",
            "/wiki/Hidemi_Hondou",
        ],
        "pisco": ["/wiki/Pisco"],
        "tequila": ["/wiki/Tequila"],
        "calvados": ["/wiki/Calvados"],
        "irish": ["/wiki/Irish"],
        "curacao": ["/wiki/Cura%C3%A7ao", "/wiki/Curacao"],
        "atsushi_miyano": ["/wiki/Atsushi_Miyano"],
        "elena_miyano": ["/wiki/Elena_Miyano"],
        "akemi_miyano": ["/wiki/Akemi_Miyano"],
        "suguru_itakura": ["/wiki/Suguru_Itakura"],
        "kiichiro_numabuchi": [
            "/wiki/Ki%27ichiro_Numabuchi",
            "/wiki/Ki'ichiro_Numabuchi",
        ],
        "renya_karasuma": ["/wiki/Renya_Karasuma"],
        "rikumichi_kusuda": ["/wiki/Rikumichi_Kusuda"],
        # 5️⃣ FBI, CIA, PSB
        "akai_subaru": ["/wiki/Shuichi_Akai", "/wiki/Subaru_Okiya"],
        "jodie": ["/wiki/Jodie_Starling"],
        "james_black": ["/wiki/James_Black"],
        "camel": ["/wiki/Andre_Camel"],
        "eisuke_hondo": ["/wiki/Eisuke_Hondou", "/wiki/Eisuke_Hondo"],
        "kazami": ["/wiki/Yuya_Kazami"],
        "scotch": ["/wiki/Hiromitsu_Morofushi"],
        "ethan_hondou": ["/wiki/Ethan_Hondou"],
        # 6️⃣ Osaka Group
        "heiji": ["/wiki/Heiji_Hattori"],
        "kazuha": ["/wiki/Kazuha_Toyama"],
        "shizuka_hattori": ["/wiki/Shizuka_Hattori"],
        "iori_moga": ["/wiki/Muga_Iori"],
        "momiji": ["/wiki/Momiji_Ooka"],
        # 7️⃣ Kaito Kid Group & Magic Kaito
        "kaito_kid": ["/wiki/Kaitou_Kid", "/wiki/Kaito_Kuroba"],
        "nakamori": ["/wiki/Ginzo_Nakamori"],
        "jirokichi": ["/wiki/Jirokichi_Suzuki"],
        "konosuke_jii": ["/wiki/Konosuke_Jii"],
        "saguru_hakuba": ["/wiki/Saguru_Hakuba"],
        "toichi_kuroba": ["/wiki/Toichi_Kuroba"],
        "aoko_nakamori": ["/wiki/Aoko_Nakamori"],
        # 8️⃣ Families & Relatives
        "yusaku_kudo": ["/wiki/Yusaku_Kudo"],
        "yukiko_kudo": ["/wiki/Yukiko_Kudo"],
        "eri_kisaki": ["/wiki/Eri_Kisaki"],
        "midori_kuriyama": ["/wiki/Midori_Kuriyama"],
        "masumi_sera": ["/wiki/Masumi_Sera"],
        "mary_sera": ["/wiki/Mary_Sera"],
        "shukichi_haneda": ["/wiki/Shukichi_Haneda"],
        "koji_haneda": ["/wiki/Kohji_Haneda", "/wiki/Koji_Haneda"],
        # 9️⃣ Secondary, Daily Life & Recurring
        "sonoko": ["/wiki/Sonoko_Suzuki"],
        "makoto_kyogoku": ["/wiki/Makoto_Kyogoku"],
        "azusa_enomoto": ["/wiki/Azusa_Enomoto"],
        "yoko_okino": ["/wiki/Yoko_Okino"],
        "araide": ["/wiki/Tomoaki_Araide"],
        "rumi_wakasa": ["/wiki/Rumi_Wakasa"],
        "sumiko_kobayashi": ["/wiki/Sumiko_Kobayashi"],
        "tsutomu_akai": ["/wiki/Tsutomu_Akai"],
        "fusae_campbell": ["/wiki/Fusae_Campbell"],
        "reiko_kujo": ["/wiki/Reiko_Kujo"],
    }

    for char in character_slugs.keys():
        df[char] = 0

    total_episodes = len(df)
    print(f"🚀 Running Extraction for ALL {total_episodes} episodes...")

    # get to the csv file and read row by row to get the episodes numbersn
    for idx, row in df.iterrows():
        ep_num = int(row["episode_no"])
        url = f"https://www.detectiveconanworld.com/wiki/Episode_{ep_num}"

        try:
            res = session.get(url, timeout=10)
            if res.status_code != 200:
                continue

            soup = BeautifulSoup(res.content, "html.parser")
            found_links = extract_cast_links(soup)

            # pass through all links and decrypt them 
            found_links_clean = [
                urllib.parse.unquote(l).lower() for l in found_links
            ]

            found_chars = []
            # char_var like "conan" : (the key) \ slugs the probabilities  
            for char_var, slugs in character_slugs.items():
                if any(slug.lower() in found_links_clean for slug in slugs):
                    df.loc[df["episode_no"] == ep_num, char_var] = 1
                    found_chars.append(char_var)

            print(
                f"  ✅ Ep {ep_num}/{total_episodes}: Found {len(found_chars)} characters",
                end="\r",
                flush=True,
            )
            time.sleep(0.12)

        except Exception as e:
            continue

    print("\n\n✅ Done processing!")
    output_file = "conan_episodes_full.csv"
    df.to_csv(output_file, index=False)

    print(
        f"\n📊 Total Taka'aki Morofushi (Komei) appearances detected: {df['takaaki_morofushi'].sum()}"
    )
    print(f"💾 File successfully saved as '{output_file}'!")


if __name__ == "__main__":
    scrape_full_dataset()