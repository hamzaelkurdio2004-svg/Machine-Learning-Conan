# requstes for getting the HTMl from the Web 
# BeatifulSoup for anlaysing the HTML content 
# re for getting the data we need from the texts 
import requests 
import pandas as pd
import re
from bs4 import BeautifulSoup

def scrape_conan_episodes():
    # URL for the episodes number 
    URLs = ["https://en.wikipedia.org/wiki/List_of_Case_Closed_episodes_(seasons_1%E2%80%9315)",
    "https://en.wikipedia.org/wiki/List_of_Case_Closed_episodes_(seasons_16%E2%80%9330)",
    "https://en.wikipedia.org/wiki/List_of_Case_Closed_episodes_(seasons_31%E2%80%93current)"]
        
    # headers for not getting blocked by the website ( it creates a fake web )
    headers = {
        "User-Agent": "ConanForecastEngine/1.0 (contact: hamza@example.com) Python/requests"
    }
    
    episodes_list = []
    
    for URL in URLs:
        page_name = URL.split("_")[-1] # just takes the last part from the url for the beauty of the code 
        print(f"⏳ connecting to the source and fetching data from {page_name}")
        try:
            response = requests.get(URL,headers=headers,timeout=10)
    
            # if the status_code doesnt equal 200 it means it failed 
            if response.status_code != 200 :
                print(f"❌ Failed to conenect to the website {response.status_code}")
                continue  # ✅ Use continue instead of return None to skip to next URL if one fails
            
            # it takes all the probabilites for any problem in the requests 
        except requests.exceptions.RequestException as e:
            print(f"❌ Connection error occurred: {e}")
            continue
    
        #transforming the HTML Data into organised creature we can get about the items in it 
        soup = BeautifulSoup(response.content , "html.parser")
    
        #Searching about the Tables inside the website that have the wikitable Name and store in the 'tables'
        tables = soup.find_all("table" , {"class" : "wikitable"})[1:]
    
    
        # getting each row in the table with tr(table_row) means it read row by row
        for table in tables:
            rows = table.find_all("tr")[1:] # skip the header row
        
            # th(table header) to find the header , Note: inside the th the text is BOLD\ td(table data) for the data
            for row in rows:
                cols = row.find_all(["th","td"])
                if len(cols) >= 7: # if its less than 4 we dont need it so we dont get IndexError
                    ep_no_raw = cols[0].text.strip() # the text of the episode number 
                    title_raw = cols[3].text.strip() # the episode title 
                    raw_date = cols[6].text.strip() # the date of the episodes 
                        
                elif len(cols) >= 6:
                    ep_no_raw = cols[0].text.strip() # the text of the episode number 
                    title_raw = cols[2].text.strip() # the episode title 
                    raw_date = cols[5].text.strip() # the date of the episodes 
                else:
                    continue
                        
                try:
                    # Clean title
                    clean_title = title_raw.split("Transliteration:")[0].strip()
                    clean_title = clean_title.split("\n")[0].replace('"', '')
                    
                    # sub(pattern,replacement,string) \D mean anything isnt a number replace it with empty text 
                    ep_clean = re.sub(r'\D','',ep_no_raw)
                    
                    # Clean air date (Extract ISO YYYY-MM-DD)
                    date_match = re.search(r'\d{4}-\d{2}-\d{2}',raw_date)
                    air_date = date_match.group(0) if date_match else raw_date
                    
                    if ep_clean :
                        episodes_list.append({
                            "episode_no":int(ep_clean),
                            "title":clean_title,
                            "air_date":air_date
                            })
                except Exception as e: # if any error happended we skip it and resume 
                    continue
                
# printing the episodes and turning it into datasets 
    df = pd.DataFrame(episodes_list)
    print(f"✅ Successfully scraped {len(df)} episodes!")
    df = df.drop_duplicates(subset=['episode_no'], keep='first').reset_index(drop=True) 
    return df

if __name__ == "__main__":
    df_episodes = scrape_conan_episodes()
    if df_episodes is not None and not df_episodes.empty:
        print("\n--- First 5 Episodes ---")
        print(df_episodes.head())
        print("\n--- Last 5 Episodes ---")
        print(df_episodes.tail())
        df_episodes.to_csv("conan_episodes_raw.csv", index=False)