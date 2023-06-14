from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import scipy.stats as stats

rows = []
for path in sorted(Path("metrics").glob("*.csv")):
    if path.name == "geoparse.csv" or path.name == "distance.csv":
        continue
    table = pd.read_csv(path)
    table = table.mask((np.abs(stats.zscore(table)) > 1).any(axis=1))
    row = table.mean().combine(table.sem(), "{0:.1f} ± {1:.1f}".format)
    row = pd.concat([pd.Series({"model": path.stem}), row], axis=0)
    rows.append(row)
print(pd.concat(rows, axis=1).T.to_latex(index=False), end="\n")

# table = pd.read_csv("metrics/distance.csv")
# table = table.quantile(np.linspace(0, 1, 11)).reset_index(names=["quantile"])
# table["quantile"] *= 100
# print(table.style.format({"quantile": "{:.0f}\%"}).hide(axis="index").to_latex())
