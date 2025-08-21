import pandas as pd
from sklearn.datasets import load_iris
from cleanit.pipeline import CleanPipeline
from cleanit.transforms import remove_duplicates, fill_missing, encode_categoricals, scale_features, report_profile

iris = load_iris(as_frame=True).frame.rename(columns={"target":"species"})
iris.loc[::15, "sepal length (cm)"] = None
iris["species"] = iris["species"].astype(str)
iris = pd.concat([iris, iris.iloc[0:5]], ignore_index=True)

pipe = CleanPipeline(steps=[
    remove_duplicates(),
    fill_missing(num_strategy="median", cat_strategy="mode"),
    encode_categoricals(method="onehot"),
    scale_features(method="standard")
])

clean = pipe.fit_transform(iris)
print(report_profile(clean))
