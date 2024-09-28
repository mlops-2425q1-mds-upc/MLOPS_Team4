from config import RAW_DATA_DIR
import pandas as pd

import texthero as hero

cols = ["sentiment", "id", "date", "query_string", "user", "text"]

df_processed = pd.read_csv(
    RAW_DATA_DIR / "training.1600000.processed.noemoticon.csv",
    header=None,
    names=cols,
    encoding="latin-1",
)

df_processed["CleanTweet"].pipe(hero.clean)
