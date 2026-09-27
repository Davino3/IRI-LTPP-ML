"""Reproduces every number, table and figure in docs/IRI_Project_Report.pdf.

Run from the repository root:  python src/report_analysis.py
All evaluation is grouped by location (sections sharing a climate record)."""
import json, warnings
warnings.filterwarnings("ignore")
import numpy as np, pandas as pd
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.model_selection import GroupKFold, GroupShuffleSplit
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler, PolynomialFeatures
from sklearn.linear_model import LinearRegression, Ridge, LogisticRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.inspection import permutation_importance
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error, precision_score, recall_score, f1_score, confusion_matrix

BLUE, ORANGE, AQUA, GREY, INK, INK2 = "#2a78d6", "#eb6834", "#1baf7a", "#898781", "#0b0b0b", "#52514e"
plt.rcParams.update({"figure.dpi": 200, "font.size": 9, "axes.spines.top": False, "axes.spines.right": False,
    "axes.grid": True, "grid.color": "#e6e5e1", "grid.linewidth": 0.6, "axes.edgecolor": GREY,
    "axes.labelcolor": INK2, "xtick.color": INK2, "ytick.color": INK2, "axes.titlesize": 10,
    "axes.titleweight": "bold", "axes.axisbelow": True, "legend.frameon": False, "font.family": "DejaVu Sans"})
R = {}

d = pd.read_csv("data/pavement_clean.csv", dtype={"SHRP_ID": str})
d = d[d.SHARE_CLASS9 <= 1].copy()
d["IS_RIGID"] = (d.PAVEMENT_TYPE == "Rigid").astype(int)
# Location group: sections that share any identical yearly climate record (same state, year,
# temperature, freezing index, freeze-thaw) sit in the same MERRA grid cell, so they share weather
# (this also joins all SPS sections at one site). Union-find over shared records.
_par = {s: s for s in d.SECTION_ID.unique()}
def _f(x):
    while _par[x] != x:
        _par[x] = _par[_par[x]]; x = _par[x]
    return x
_key = list(zip(d.STATE_CODE, d.YEAR, d.TEMP_AVG.round(2), d.FREEZE_INDEX, d.FREEZE_THAW))
for _k, _g in d.assign(_K=_key).groupby("_K").SECTION_ID:
    _s = _g.unique()
    for _x in _s[1:]: _par[_f(_x)] = _f(_s[0])
d["SITE_ID"] = d.SECTION_ID.map(_f)
d["SPS_SITE"] = np.where(d.GPS_SPS == "S", d.STATE_CODE.astype(str).str.zfill(2) + "_" + d.SHRP_ID.str[:2], d.SECTION_ID)
d = d.sort_values(["SECTION_ID", "CONSTRUCTION_NO", "YEAR"])
g = d.groupby(["SECTION_ID", "CONSTRUCTION_NO"])
d["PREV_IRI"] = g.IRI_AVG.shift(1); d["GAP_YRS"] = d.YEAR - g.YEAR.shift(1)
d["IRI_CHANGE"] = d.IRI_AVG - g.IRI_AVG.transform("first")
R["n_rows"] = len(d); R["n_sections"] = d.SECTION_ID.nunique(); R["n_sites"] = d.SITE_ID.nunique()
R["n_states"] = d.STATE_CODE_EXP.nunique(); R["years"] = [int(d.YEAR.min()), int(d.YEAR.max())]
R["gps_sections"] = int(d[d.GPS_SPS=="G"].SECTION_ID.nunique()); R["sps_sections"] = int(d[d.GPS_SPS=="S"].SECTION_ID.nunique())
R["sps_sites"] = int(d[d.GPS_SPS=="S"].SPS_SITE.nunique()); R["sps_rule_groups"] = int(d.SPS_SITE.nunique())
R["iri"] = d.IRI_AVG.describe().round(3).to_dict()
R["skew"] = [round(d.IRI_AVG.skew(), 2), round(np.log(d.IRI_AVG).skew(), 2)]
cond = pd.cut(d.IRI_AVG, [0, 1.5, 2.68, np.inf], labels=["Good", "Fair", "Poor"])
R["cond"] = cond.value_counts(normalize=True).round(3).to_dict()
R["rigid_median"] = d.groupby("PAVEMENT_TYPE").IRI_AVG.median().round(2).to_dict()
sec_mean = d.groupby("SECTION_ID").IRI_AVG.transform("mean")
R["between_share"] = round(((sec_mean - d.IRI_AVG.mean())**2).sum() / ((d.IRI_AVG - d.IRI_AVG.mean())**2).sum(), 3)

F = ["AGE_SINCE_CN", "IS_REHAB", "IS_RIGID", "URBAN", "LOG_ESAL", "LOG_AADTT", "SHARE_CLASS9", "TEMP_AVG", "FREEZE_INDEX", "FREEZE_THAW"]
FB = F + ["PREV_IRI", "GAP_YRS"]
R["corr"] = d[F + ["IRI_AVG"]].corr(method="spearman")["IRI_AVG"].drop("IRI_AVG").round(3).to_dict()

L = d.dropna(subset=["PREV_IRI"]).copy()
R["n_lag"] = len(L); R["n_lag_sites"] = L.SITE_ID.nunique(); R["n_lag_sections"] = L.SECTION_ID.nunique()
y = L.IRI_AVG.to_numpy(); sites = L.SITE_ID.to_numpy(); secs = L.SECTION_ID.to_numpy()
XA, XB = L[F].to_numpy(), L[FB].to_numpy(); prev = L.PREV_IRI.to_numpy()
cv = GroupKFold(5)

def rf(): return RandomForestRegressor(n_estimators=300, min_samples_leaf=5, n_jobs=-1, random_state=42)
models = {
 "Mean of training IRI": ("mean", None),
 "Linear regression (A)": (lambda: make_pipeline(StandardScaler(), LinearRegression()), XA),
 "Polynomial degree 2 + ridge (A)": (lambda: make_pipeline(StandardScaler(), PolynomialFeatures(2, include_bias=False), StandardScaler(), Ridge(alpha=1000)), XA),
 "Random forest (A)": (rf, XA),
 "Persistence (previous IRI)": ("persist", None),
 "Linear regression (B)": (lambda: make_pipeline(StandardScaler(), LinearRegression()), XB),
 "Random forest (B)": (rf, XB),
}
def cv_eval(groups):
    out = {}
    for name, (mk, X) in models.items():
        r2s, rm, ma = [], [], []
        for tr, te in cv.split(L, y, groups):
            if mk == "mean": p = np.full(len(te), y[tr].mean())
            elif mk == "persist": p = prev[te]
            else: p = mk().fit(X[tr], y[tr]).predict(X[te])
            r2s.append(r2_score(y[te], p)); rm.append(mean_squared_error(y[te], p)**.5); ma.append(mean_absolute_error(y[te], p))
        out[name] = dict(r2=round(np.mean(r2s), 3), r2_sd=round(np.std(r2s), 3), rmse=round(np.mean(rm), 3), mae=round(np.mean(ma), 3))
        print(name, out[name])
    return out
R["cv_site"] = cv_eval(sites)
print("--- by section")
R["cv_section"] = {k: v for k, v in cv_eval(secs).items() if k in ("Linear regression (A)", "Random forest (A)", "Linear regression (B)", "Random forest (B)", "Persistence (previous IRI)")}
print("--- by SPS site rule")
_sps = L.SPS_SITE.to_numpy()
R["cv_spsrule"] = {k: v for k, v in cv_eval(_sps).items() if k in ("Linear regression (A)", "Random forest (A)")}

# change model
ch = y - prev; c2 = []
for tr, te in cv.split(L, y, sites):
    c2.append(r2_score(ch[te], make_pipeline(StandardScaler(), LinearRegression()).fit(XB[tr], ch[tr]).predict(XB[te])))
R["change_r2"] = round(np.mean(c2), 3)

# polynomial degree, site CV, feature set A on full data (not lagged)
yf = d.IRI_AVG.to_numpy(); Xf = d[F].to_numpy(); sf = d.SITE_ID.to_numpy()
R["poly"] = {}
for deg in [1, 2, 3, 4]:
    tr_s, te_s = [], []
    for tr, te in cv.split(d, yf, sf):
        m = make_pipeline(StandardScaler(), PolynomialFeatures(deg, include_bias=False), StandardScaler(), LinearRegression()).fit(Xf[tr], yf[tr])
        tr_s.append(r2_score(yf[tr], m.predict(Xf[tr]))); te_s.append(r2_score(yf[te], m.predict(Xf[te])))
    R["poly"][deg] = [round(np.mean(tr_s), 3), round(np.median(te_s), 3), round(np.min(te_s), 3)]
    print("deg", deg, R["poly"][deg])
R["ridge"] = {}
for lam in [1, 100, 1000, 10000]:
    s = []
    for tr, te in cv.split(d, yf, sf):
        m = make_pipeline(StandardScaler(), PolynomialFeatures(3, include_bias=False), StandardScaler(), Ridge(alpha=lam)).fit(Xf[tr], yf[tr])
        s.append(r2_score(yf[te], m.predict(Xf[te])))
    R["ridge"][lam] = round(np.mean(s), 3); print("ridge", lam, R["ridge"][lam])

# held-out site split for figures + classification + coefficients
tr, te = next(GroupShuffleSplit(1, test_size=0.2, random_state=42).split(L, y, sites))
R["holdout"] = dict(n_train=len(tr), n_test=len(te), sites_test=int(len(set(sites[te]))))
linA = make_pipeline(StandardScaler(), LinearRegression()).fit(XA[tr], y[tr]); pA = linA.predict(XA[te])
linB = make_pipeline(StandardScaler(), LinearRegression()).fit(XB[tr], y[tr]); pB = linB.predict(XB[te])
R["coef_B"] = pd.Series(linB[-1].coef_, FB).round(4).to_dict()
R["coef_A"] = pd.Series(linA[-1].coef_, F).round(4).to_dict()
forB = rf().fit(XB[tr], y[tr])
pi = permutation_importance(forB, XB[te], y[te], n_repeats=10, random_state=42)
R["perm"] = pd.Series(pi.importances_mean, FB).round(4).to_dict()

# classification
yp = (y > 2.68).astype(int); yt = yp[te]
def cm(p):
    tn, fp, fn, tp = confusion_matrix(yt, p).ravel()
    return dict(prec=round(precision_score(yt, p, zero_division=0), 3), rec=round(recall_score(yt, p), 3), f1=round(f1_score(yt, p), 3), acc=round((tp+tn)/len(yt), 3), TP=int(tp), FN=int(fn), FP=int(fp))
clf = make_pipeline(StandardScaler(), LogisticRegression(max_iter=1000)).fit(XB[tr], yp[tr]); pr = clf.predict_proba(XB[te])[:, 1]
clfb = make_pipeline(StandardScaler(), LogisticRegression(class_weight="balanced", max_iter=1000)).fit(XB[tr], yp[tr])
clfA = make_pipeline(StandardScaler(), LogisticRegression(class_weight="balanced", max_iter=1000)).fit(XA[tr], yp[tr])
R["cls"] = {"Always not poor": cm(np.zeros(len(yt), int)), "Persistence": cm((prev[te] > 2.68).astype(int)),
            "Logistic, threshold 0.5": cm((pr >= .5).astype(int)), "Logistic, threshold 0.3": cm((pr >= .3).astype(int)),
            "Logistic, threshold 0.1": cm((pr >= .1).astype(int)), "Logistic, balanced weights": cm(clfb.predict(XB[te])),
            "Logistic without previous IRI (balanced)": cm(clfA.predict(XA[te]))}
R["cls_poor_test"] = int(yt.sum()); R["poor_share"] = round(yp.mean(), 3)
R["logit_coef"] = pd.Series(clf[-1].coef_[0], FB).round(3).to_dict()
for k, v in R["cls"].items(): print(k, v)

# ---------------- figures
W1 = 6.3
fig, ax = plt.subplots(figsize=(W1, 2.8))
ax.hist(d.IRI_AVG, bins=70, color=BLUE, edgecolor="white", linewidth=0.3)
for x, lab in [(1.5, "Good | Fair  1.50"), (2.68, "Fair | Poor  2.68")]:
    ax.axvline(x, color=INK2, ls="--", lw=0.9); ax.text(x + 0.05, ax.get_ylim()[1] * 0.92, lab, color=INK2, fontsize=7.5)
ax.set(xlabel="IRI (m/km)", ylabel="Number of measurements", xlim=(0, 6))
fig.tight_layout(); fig.savefig("figures/fig1_iri_distribution.png"); plt.close(fig)

fig, axes = plt.subplots(1, 2, figsize=(W1, 2.7))
med = d.groupby("AGE_SINCE_CN").IRI_AVG.median(); cnt = d.AGE_SINCE_CN.value_counts()
ok = cnt[cnt >= 100].index
axes[0].plot(med.loc[med.index.isin(ok)].index, med.loc[med.index.isin(ok)].values, color=BLUE, lw=2, marker="o", ms=3)
axes[0].set(title="(a) Median IRI by age", xlabel="Years since construction event", ylabel="Median IRI (m/km)", ylim=(0.9, 1.5))
chg = d.groupby("AGE_SINCE_CN").IRI_CHANGE.median()
axes[1].plot(chg.loc[chg.index.isin(ok)].index, chg.loc[chg.index.isin(ok)].values, color=ORANGE, lw=2, marker="o", ms=3)
axes[1].axhline(0, color=GREY, lw=0.8, ls="--")
axes[1].set(title="(b) Change since first survey", xlabel="Years since construction event", ylabel="Change in IRI (m/km)")
fig.tight_layout(); fig.savefig("figures/fig2_age_trend.png"); plt.close(fig)

fig, axes = plt.subplots(1, 2, figsize=(W1, 3.0), sharex=True, sharey=True)
for ax, p, t, c in [(axes[0], pA, f"(a) Feature set A: R² = {r2_score(y[te], pA):.2f}", BLUE), (axes[1], pB, f"(b) Feature set B: R² = {r2_score(y[te], pB):.2f}", ORANGE)]:
    ax.scatter(y[te], p, s=3, alpha=0.3, color=c, linewidths=0)
    ax.plot([0, 5.5], [0, 5.5], color=INK2, ls="--", lw=0.9)
    ax.set(title=t, xlabel="Measured IRI (m/km)", xlim=(0, 5.5), ylim=(0, 5.5))
axes[0].set_ylabel("Predicted IRI (m/km)")
fig.tight_layout(); fig.savefig("figures/fig3_predicted_vs_measured.png"); plt.close(fig)
R["holdout_r2"] = [round(r2_score(y[te], pA), 3), round(r2_score(y[te], pB), 3), round(r2_score(y[te], prev[te]), 3)]

imp = pd.Series(R["perm"]).drop("PREV_IRI").sort_values()
fig, ax = plt.subplots(figsize=(W1, 2.9))
ax.barh(imp.index, imp.values, color=BLUE, height=0.6)
ax.set(xlabel="Mean drop in test R² when the feature is shuffled")
ax.axvline(0, color=GREY, lw=0.8)
fig.tight_layout(); fig.savefig("figures/fig4_permutation_importance.png"); plt.close(fig)

ths = np.linspace(0.02, 0.9, 45)
P = [precision_score(yt, (pr >= t).astype(int), zero_division=0) for t in ths]
Rc = [recall_score(yt, (pr >= t).astype(int)) for t in ths]
fig, ax = plt.subplots(figsize=(W1, 2.7))
ax.plot(ths, P, color=BLUE, lw=2, label="Precision"); ax.plot(ths, Rc, color=ORANGE, lw=2, label="Recall")
ax.axvline(0.5, color=GREY, ls="--", lw=0.8); ax.text(0.51, 0.05, "default 0.5", color=INK2, fontsize=7.5)
ax.set(xlabel="Probability threshold for flagging a section as poor", ylabel="Score", ylim=(0, 1.02))
ax.legend(loc="center right")
fig.tight_layout(); fig.savefig("figures/fig5_precision_recall_threshold.png"); plt.close(fig)

json.dump(R, open("results/report_results.json", "w"), indent=1, default=str)
print(json.dumps({k: R[k] for k in ["n_rows","n_sections","n_sites","n_lag","n_lag_sites","holdout","holdout_r2","change_r2","perm","coef_B","sps_sections","gps_sections","sps_sites","between_share","corr"]}, indent=1, default=str))
