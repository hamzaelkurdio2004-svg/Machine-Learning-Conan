import requests
import pandas as pd
from bs4 import BeautifulSoup

def add_canon_status():
    # reading the file 
    print("⏳ Reading Raw episodes from 'conan_episodes_raw.csv'...")
    try:
        df = pd.read_csv("conan_episodes_raw.csv")
    except FileNotFoundError:
        print("❌ File 'conan_episodes_raw.csv' not found! Make sure you ran conan1.py first.")
        return
    
    URL = "https://www.animefillerlist.com/shows/detective-conan"
    headers = {
        "User-Agent": "ConanForecastEngine/1.0 (contact: hamza@example.com) Python/requests"
    }
    
    print("⏳ Fetching Filler or Manga data from AnimeFillerList...")
    try:
        response = requests.get(URL,headers=headers,timeout=10)
        if response.status_code !=200:
            print(f"❌ Failed to connect to AnimeFillerList. Status code: {response.status_code}")
            return
    except Exception as e:
        print(f"❌ Connection error: {e}")
        return
    
    soup = BeautifulSoup(response.content , "html.parser")
    table = soup.find("table" , {"class":"EpisodeList"})
    
    canon_mapping = {}
    
    if table:
        rows = table.find_all("tr")[1:]
        for row in rows:
            cols = row.find_all("td")
            if len(cols) >=3 :
                try:
                    ep_no = int(cols[0].text.strip())
                    ep_type = cols[2].text.strip().lower()
                    
                    if "manga" in ep_type:
                        is_canon = 1
                    else:
                        is_canon = 0
                        
                    canon_mapping[ep_no] = is_canon
                except ValueError:
                    continue
    
    # creates a new column named is canon by comparing the episodes numbers with each other            
    df["is_canon"] = df["episode_no"].map(canon_mapping).fillna(0).astype(int)
    
    canon_count = (df["is_canon"] == 1).sum()
    filler_count = (df["is_canon"] == 0).sum()
    
    print("\n✅ Processing Complete!")
    print(f"📊 Summary: {canon_count} Manga Canon episodes | {filler_count} Filler episodes.")
    print("\n--- First 10 Episodes preview ---")
    print(df[['episode_no', 'title', 'air_date', 'is_canon']].head(10))
    
    df.to_csv("conan_episodes_clean.csv", index=False)
    print("\n💾 Saved output file as 'conan_episodes_clean.csv'!")
    
if __name__ == "__main__":
    add_canon_status()