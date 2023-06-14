import pandas as pd
import requests
import spacy
from spacy.language import Language


def geoparse(
    text: str,
    ner: Language = spacy.load("pt_core_news_sm"),
) -> tuple[float, float]:
    geocode = ", ".join({ent.text for ent in ner(text).ents if ent.label_ == "LOC"})
    response = requests.get(
        "https://nominatim.openstreetmap.org/search",
        {"q": geocode, "addressdetails": 1, "format": "json", "limit": 1},
    ).json()
    if not response:
        return pd.NA, pd.NA
    return float(response[0]["lat"]), float(response[0]["lon"])


if __name__ == "__main__":
    data = pd.read_excel("data/news.xlsx")
    data = data[data["latitude"].notna()]
    data["~latitude"], data["~longitude"] = zip(*data["text"].apply(geoparse))
    data["-latitude"] = abs(data["latitude"] - data["~latitude"])
    data["-longitude"] = abs(data["longitude"] - data["~longitude"])
    data[["-latitude", "-longitude"]].to_csv("metrics/distance.csv", index=False)
