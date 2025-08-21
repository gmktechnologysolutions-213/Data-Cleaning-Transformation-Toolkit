import pandas as pd
from cleanit.transforms import remove_duplicates, fill_missing, encode_categoricals, scale_features, report_profile

def test_remove_duplicates():
    df = pd.DataFrame({"id":[1,1,2], "v":[10,10,20]})
    out = remove_duplicates(subset=["id"]).transform(df)
    assert len(out)==2

def test_fill_missing():
    df = pd.DataFrame({"a":[1.0, None, 3.0], "b":["x", None, "y"]})
    fm = fill_missing(num_strategy="mean", cat_strategy="constant", cat_constant="NA").fit(df)
    out = fm.transform(df)
    assert out["a"].isna().sum()==0 and out["b"].isna().sum()==0

def test_encode_onehot():
    df = pd.DataFrame({"color":["red","blue","red"], "v":[1,2,3]})
    enc = encode_categoricals(method="onehot").fit(df)
    out = enc.transform(df)
    assert set(c for c in out.columns if c.startswith("color_")) == set(["color_blue","color_red"])

def test_scale():
    df = pd.DataFrame({"x":[0,5,10]})
    sc = scale_features(method="minmax").fit(df)
    out = sc.transform(df)
    assert out["x"].min() == 0 and out["x"].max() == 1

def test_report_profile():
    df = pd.DataFrame({"x":[1,2,None]})
    r = report_profile(df)
    assert "missing_%" in r.columns
