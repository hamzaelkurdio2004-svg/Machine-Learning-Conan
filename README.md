# Detective Conan Episode Forecast

A small data project that forecasts the **next 100 episodes of Detective Conan**: which episodes are likely to be manga canon or filler, who appears in each, and when each of 89 characters is most likely to show up next. The results are served through a single-file web app.

> Fan project, not affiliated with or endorsed by the creators, studios or broadcasters of Detective Conan.

## What it does

- **Outlook for the next 100 episodes.** Press Enter and the app shows the expected number of manga episodes, filler episodes, and the average cast size per episode.
- **Character files.** Search any of 89 characters to open a profile card with:
  - status: alive, deceased, or presumed dead with a doubtful fate (Akemi Miyano)
  - number of episodes they appeared in, and the last episode they were seen in
  - the most likely window (up to 5 episodes) for their next appearance, with its probability
  - the chance of appearing at least once in the next 100 episodes
  - a portrait and a custom tone, which you provide yourself
- **A typing guide** above the search box explains how to write names (`kaito_kid`, `akai_subaru`) and lists every name.

## How it works

```
conan_data.py   ->  conan_episodes_raw.csv     episode number, title, air date
conan_manga.py  ->  conan_episodes_clean.csv   adds is_canon (1 = manga, 0 = filler)
conan_char.py   ->  conan_episodes_full.csv    adds one 0/1 column per character
conanml_v2.py   ->  conan_future_100_episodes.csv
                    conan_future_probs.csv
                    conan_character_cards.csv  forecasts
build_ui.py     ->  conan_ui.html              the web app (data embedded)
```

**Forecasting method (`conanml_v2.py`)**

1. **Manga or filler:** the schedule is simulated from the lengths of recent manga and filler runs, because the show airs in blocks (manga arcs of 2 to 4 episodes between filler runs).
2. **Characters:** for each character a gradient boosting model predicts the probability of appearing, using features such as episodes since last appearance, rolling appearance rates, the character's rate in manga versus filler episodes, whether the episode is manga, and recent activity of related characters.
3. **Monte Carlo simulation:** appearances are sampled from those probabilities, fed back as history, and repeated 200 times. Averaging gives each episode's probabilities, and the distribution of the first appearance gives the "next appearance window".
4. Characters who have never appeared in a filler episode are forced to 0 in filler episodes. Dead characters are not forecast.

**Why probabilities and not 0/1 predictions?** An early version fed its own hard yes/no predictions back in as history. After a few steps it collapsed to the majority answer: every episode was filler with the same cast. Sampling from probabilities keeps the uncertainty alive.

## Results and limits

Backtests predict 100 episodes ahead, the same way the model is used:

| Task | Model | Simple baseline |
|---|---|---|
| Manga vs filler (Brier score, lower is better) | 0.232 | 0.235 |
| Character appearances (ROC-AUC) | 0.846 and 0.864 | 0.849 and 0.863 |
| Character appearances if the true manga/filler schedule were known | about 0.91 | not applicable |

The model is **only slightly better than a baseline** that uses recent averages. Forecasting 100 episodes ahead is inherently uncertain, and the biggest gain would come from knowing the real manga/filler schedule. Treat the output as probabilities, not predictions.

## Getting started

Requires Python 3.9 or newer.

```bash
pip install requests pandas beautifulsoup4 scikit-learn numpy
```

Run the scripts in order, from the project folder:

```bash
python conan_data.py     # episode list from Wikipedia
python conan_manga.py    # manga/filler labels from AnimeFillerList
python conan_char.py     # cast per episode (about 1,200 requests, takes several minutes)
python conanml_v2.py     # backtests, forecasts and exports (about a minute)
python build_ui.py       # builds conan_ui.html
```

Then open `conan_ui.html` in a browser. `build_ui.py` needs `ui_template.html` in the same folder.

The scrapers send a `User-Agent` header with a placeholder email. Replace it with your own contact before running them, and keep the request delays.

## Adding portraits and tones

Images and audio are not included. Add your own next to `conan_ui.html`, named after the character key shown in the card (lowercase):

```
assets/
  images/heiji.jpg        (jpg, png or webp)
  audio/heiji.mp3
```

Characters without files show a monogram placeholder and a "no tone yet" note.

## Deploying

The app is static, so you can host the folder anywhere. Rename `conan_ui.html` to `index.html`, keep the `assets` folder beside it, and upload the folder to Netlify Drop or GitHub Pages.

## Project files

| File | Purpose |
|---|---|
| `conan_data.py` | Scrapes episode numbers, titles and air dates |
| `conan_manga.py` | Labels each episode as manga canon or filler |
| `conan_char.py` | Detects which of 89 characters appear in each episode |
| `conanml_v2.py` | Backtests, simulation and forecast exports |
| `build_ui.py`, `ui_template.html` | Builds the web app |
| `conanml.py` | First version, kept for reference (superseded by `conanml_v2.py`) |

## Data sources

- [Wikipedia](https://en.wikipedia.org/wiki/List_of_Case_Closed_episodes_(seasons_1%E2%80%9315)): episode list and air dates
- [AnimeFillerList](https://www.animefillerlist.com/shows/detective-conan): manga vs filler
- [Detective Conan World](https://www.detectiveconanworld.com): cast per episode

Data comes from these sites' public pages and belongs to their respective authors. Please follow their terms of use if you rerun the scrapers.

## Known issues

- Two characters (`irish`, `curacao`) have no appearances in the data, probably because their page links in `conan_char.py` don't match the site. Check them if you rely on those cards.
- Dead characters are not forecast, even though some appear in flashbacks.

## License

Add a license of your choice before publishing (for example MIT for the code). It does not cover any images, audio or other material you add from the series.
