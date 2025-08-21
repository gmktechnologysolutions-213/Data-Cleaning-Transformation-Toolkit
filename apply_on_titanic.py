import pandas as pd
from cleanit.pipeline import CleanPipeline
from cleanit.transforms import remove_duplicates, fill_missing, encode_categoricals, scale_features, report_profile

df = pd.read_csv("data/titanic_like.csv")
pipe = CleanPipeline(steps=[
    remove_duplicates(subset=["PassengerId"]),
    fill_missing(num_strategy="median", cat_strategy="mode"),
    encode_categoricals(method="onehot"),
    scale_features(method="robust")
])

clean = pipe.fit_transform(df)
print(report_profile(clean))
