import re
from pathlib import Path

import pandas as pd
import requests
from tqdm import tqdm

if not Path("data/fires.xlsx").exists():
    fires = pd.read_excel(
        "data/ref_fire_face_v3_102019.xlsx",
        sheet_name="id_evento",
        names=["id", "fire"],
        usecols=["id", "fire"],
    )
    regex = re.compile(Path("data/fires.re").read_text())
    fires["geocode"] = fires["fire"].apply(lambda geocode: regex.sub("", geocode))
    fires.to_excel("data/fires.xlsx", index=False)
    exit(0)

fires = pd.read_excel("data/fires.xlsx", usecols=["id", "geocode"]) 
for index, geocode in enumerate(tqdm(fires["geocode"])):
    response = requests.get(
        "https://nominatim.openstreetmap.org/search",
        {"q": geocode, "format": "json", "limit": 1},
    ).json()
    if not response:
        continue
    fires.at[index, "latitude"] = response[0]["lat"]
    fires.at[index, "longitude"] = response[0]["lon"]
news = pd.read_excel(
    "data/ref_fire_face_v3_102019.xlsx",
    sheet_name="referencia",
    names=["", "text", "", "", "", "related", "fire", "id"],
    usecols=["text", "related", "fire", "id"],
).drop_duplicates()
news["related"] += news["fire"]
merged = news.join(fires.set_index("id"), on="id").drop(columns=["id", "geocode"])
merged.to_excel("data/news.xlsx", index=False)
