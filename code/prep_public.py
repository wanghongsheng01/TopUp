"""Convert a public dataset (CFPB or Amazon Reviews 2023) into the replay's table layout.

Input : public/<ds>/<cur>.parquet and <hist>.parquet with key, ts, text, gold, area, market, cat1
        public/<ds>/classes.json  {"labels": [...], "question": "..."}  (written by hand from the dataset's STATUS.md)
Output: public/<ds>/prepared.parquet and prepared_hist.parquet with the columns the labeler and the replay read
        (memo_deid holds the public text; edit_day = -1 because these sources have no edit events)
Usage : python prep_public.py cfpb cfpb_2023q4 cfpb_2022q4
"""
import os, sys
import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
ds, cur, hist = sys.argv[1:4]
D = os.path.join(HERE, "..", "public", ds)
c = pd.read_parquet(os.path.join(D, cur + ".parquet")); h = pd.read_parquet(os.path.join(D, hist + ".parquet"))
t0 = pd.Timestamp(c.ts.min()).normalize()
for df, name in ((c, "prepared.parquet"), (h, "prepared_hist.parquet")):
    out = pd.DataFrame({
        "key": df.key.astype(str).values,
        "day": ((pd.to_datetime(df.ts).dt.normalize() - t0).dt.days).astype(int).values,
        "edit_day": -1,
        "area_name": df.area.astype(str).values, "market_name": df.market.astype(str).values,
        "category_lv1_name": df.cat1.astype(str).values,
        "code_lv1": -1,
        "gold": df.gold.astype(int).values,
        "memo_deid": df.text.astype(str).str.replace(r"\s+", " ", regex=True).str.slice(0, 1500).values})
    out.to_parquet(os.path.join(D, name), index=False)
    print(name, len(out), "days", out.day.min(), out.day.max(), "classes", out.gold.value_counts().sort_index().to_dict())
