import json
import sys
from itertools import chain
from operator import attrgetter
from pathlib import Path
from typing import Optional

import geopandas as gpd
import matplotlib.pyplot as plt
import requests
import spacy
from matplotlib_scalebar.scalebar import ScaleBar
from shapely.geometry import shape as shapify
from shapely.geometry.base import BaseGeometry
from shapely.geometry.point import Point
from snscrape.modules.twitter import TwitterSearchScraper
from spacy.language import Language
from tqdm import tqdm
from transformers import AutoModelForSequenceClassification, AutoTokenizer, pipeline


def classify(
    text: str,
    classifier=pipeline(
        "text-classification",
        model=AutoModelForSequenceClassification.from_pretrained(
            "./models/fire-neuralmind-bert-large-portuguese-cased"
        ),
        tokenizer=AutoTokenizer.from_pretrained(
            "neuralmind/bert-large-portuguese-cased"
        ),
    ),
) -> bool:
    predictions = classifier(text)[0]
    return predictions["label"] == "LABEL_0"


def geoparse(
    text: str,
    ner: Language = spacy.load("pt_core_news_sm"),
) -> Optional[BaseGeometry]:
    geocode = ", ".join({ent.text for ent in ner(text).ents if ent.label_ == "LOC"})
    response = requests.get(
        "https://nominatim.openstreetmap.org/search",
        {
            "q": geocode,
            "addressdetails": 1,
            "format": "json",
            "polygon_geojson": 1,
            "limit": 1,
        },
    ).json()
    return shapify(response[0]["geojson"]) if response else None


if __name__ == "__main__":
    regions = gpd.read_file("data/districts.geojson")

    keywords = ["incêndio"]
    options = {
        "-filter": "replies",
        "lang": "pt",
        "until": sys.argv[1],
    }
    volume = 100
    query = " ".join(chain(keywords, map("{0[0]}:{0[1]}".format, options.items())))
    tweets = map(attrgetter("rawContent"), TwitterSearchScraper(query).get_items())
    regions["reports"] = [[] for _ in range(len(regions))]
    with tqdm(total=volume) as bar:
        for tweet in tweets:
            if not classify(tweet):
                continue
            if not (geometry := geoparse(tweet)):
                continue
            included = False
            for index, region in regions.iterrows():
                if geometry.intersects(region["geometry"]):
                    intersection = geometry.intersection(region["geometry"])
                    if intersection.area > 1e-5 or type(intersection) is Point:
                        regions.at[index, "reports"].append(tweet)
                        included = True
            if not included:
                continue
            bar.update()
            if (volume := volume - 1) == 0:
                break
    Path("data/reports.json").write_text(json.dumps(regions["reports"].tolist()))

    # regions["reports"] = json.loads(Path("data/reports.json").read_text())

    regions["count"] = list(map(len, regions["reports"]))
    plt.imshow(
        plt.imread("img/north.png"),
        extent=[-1.06e6, -1.06e6 + 0.05375e6, 4.95e6, 5.05e6],
    )
    regions.to_crs(3395).plot(
        ax=plt.gca(),
        column="count",
        cmap="Oranges",
        edgecolor="black",
        linewidth=0.5,
        legend=True,
        legend_kwds={"shrink": 0.5, "pad": -0.05, "anchor": (0, 0.3)},
    ).add_artist(ScaleBar(1, location="lower right", pad=0.6))
    plt.axis("off")
    plt.savefig(f"img/heatmap-until-{sys.argv[1]}.png", dpi=300, bbox_inches="tight")
