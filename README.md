# Predicting Pavement Roughness (IRI) from Traffic and Climate

Regression and classification models of the International Roughness Index (IRI) on US Long-Term Pavement Performance (LTPP) data, with honest, leakage-aware validation.

**Author:** David Badu, Department of Civil Engineering, KNUST, Kumasi, Ghana

---

## Why this project

The International Roughness Index (IRI, in m/km) is the standard measure of how rough a road is, and it drives maintenance decisions. This project asks two questions:

1. How much of a pavement's roughness can traffic, climate and section history explain?
2. How much depends on simply knowing the pavement's earlier condition?

I built the project to apply everything from the *Supervised Machine Learning: Regression and Classification* course to a real pavement engineering dataset:

- linear regression written from scratch with gradient descent,
- feature scaling,
- polynomial features and regularisation,
- logistic regression.

I then went beyond the course with random forests, grouped cross-validation and checks for data leakage.

## Key findings

All numbers below come from the notebook (`notebooks/iri_regression_classification.ipynb`). Models were trained on 80% of road sections and tested on the 20% they never saw.

| Model (test set, unseen sections) | R² | RMSE (m/km) |
|---|---|---|
| Predict the mean IRI | −0.008 | 0.619 |
| Linear regression, traffic + climate + section attributes | 0.068 | 0.595 |
| Polynomial (degree 3) + ridge, same features | 0.069 | 0.595 |
| **Persistence:** next IRI = previous IRI | 0.897 | 0.197 |
| **Linear regression with previous IRI** | **0.914** | **0.179** |

1. **Traffic and climate alone explain very little roughness.** Test R² was 0.068, only about 4% better than guessing the mean. Training R² (0.100) was also low, so the model underfits. Polynomial features did not help: degree 4 reached training R² 0.364 but test R² −1.87 (overfitting), and ridge regularisation only restored 0.069.
2. **Previous condition dominates.** Simply repeating the last measured IRI gives R² = 0.897. Adding the other features raises this to 0.914 and cuts the error by about 9% (0.197 to 0.179 m/km), but explains only 5.6% of the change in IRI between surveys. After previous IRI, the most useful features were years since the last survey, freezing index and the share of heavy (Class 9) trucks.
3. **The validation design changes the conclusions.** In five-fold cross-validation, a random forest without previous IRI scored R² 0.345 when folds were grouped by section, but only 0.092 when grouped by LTPP site. Sections at the same site share traffic and weather, so the forest had been recognising sites it had already seen. With previous IRI, both groupings agree (about 0.92).
4. **Classification of poor pavements** (IRI > 2.68 m/km; 97 of 2,295 test observations): always predicting "not poor" is 95.8% accurate but finds no poor roads. Logistic regression with previous IRI found 65% of poor observations at 88% precision (persistence: 61% at 92%). Lowering the threshold to 0.3 found 80% at 77% precision. Without previous IRI, only 7% of its alerts were correct.

<p align="center">
  <img src="figures/notebook_predicted_vs_actual.png" width="460"><br>
  <em>From the notebook: predicted against actual IRI on the test set using traffic and climate only. Predictions bunch around the average.</em>
</p>

> **Note on the report.** `docs/IRI_Project_Report.pdf` repeats the analysis with an even stricter test (holding out whole locations that share a climate record, via `src/report_analysis.py`). Its numbers are slightly lower (for example, the random forest without previous IRI falls to about zero) but the conclusions are the same.

## Repository contents

```
IRI-LTPP-ML/
├── data/
│   └── pavement_clean.csv          cleaned dataset (15,940 rows, 1,550 sections, 1989 to 2025)
├── notebooks/
│   └── iri_regression_classification.ipynb   the full step-by-step analysis (Steps 1 to 11)
├── src/
│   ├── clean_data.py               builds pavement_clean.csv from the raw LTPP exports
│   └── report_analysis.py          reproduces every number and figure in the report
├── figures/                        figures used in the README and the report
├── results/
│   └── report_results.json         all numbers reported in the report
├── docs/
│   ├── IRI_Project_Report.pdf                      the write-up (11 pages)
│   ├── IRI_Project_Code_Guide.pdf                  every notebook cell explained, with outputs
│   └── Machine_Learning_from_Zero_IRI_Project.pdf  the whole project explained for beginners
├── requirements.txt
└── LICENSE
```

## How to run

```bash
git clone https://github.com/Davino3/IRI-LTPP-ML.git
cd IRI-LTPP-ML
pip install -r requirements.txt

# the step-by-step notebook
jupyter notebook notebooks/iri_regression_classification.ipynb

# reproduce the report's tables and figures (takes a few minutes)
python src/report_analysis.py
```

Tested with Python 3.11 and 3.14. Small differences in the third decimal place can appear between library versions.

## Data

- **Source:** US Federal Highway Administration, LTPP InfoPave (https://infopave.fhwa.dot.gov), extracted in September 2026.
- **Tables used:** ANALYSIS_IRI, TRF_TREND, EXPERIMENT_SECTION, SHRP_INFO and the MERRA climate tables.
- **Repository contents:** only the cleaned file is included. The raw exports can be requested from InfoPave, and `src/clean_data.py` rebuilds the cleaned file from them.

| Column | Meaning |
|---|---|
| IRI_AVG | Target: mean IRI of the visit (m/km) |
| AGE_SINCE_CN | Years since the current construction event began |
| IS_REHAB, PAVEMENT_TYPE, URBAN | Rehabilitated, flexible or rigid, urban or rural |
| LOG_ESAL, LOG_AADTT, SHARE_CLASS9 | Traffic loading, trucks per day, heavy-truck share |
| TEMP_AVG, FREEZE_INDEX, FREEZE_THAW | Annual climate |
| SECTION_ID | State code + SHRP_ID |

## Methods in brief

- **Two feature sets.**
  - A: traffic, climate and section attributes (10 features).
  - B: set A plus previous IRI and years since the previous survey.
- **Models.**
  - Linear regression, written from scratch with gradient descent and checked against scikit-learn.
  - Polynomial ridge regression.
  - Decision trees and random forests.
  - Logistic regression.
- **Validation.**
  - GroupKFold cross-validation so that related data never appears in both training and test folds.
  - The notebook groups by section and by LTPP site.
  - The report groups by location (sections sharing the same climate record), the strictest option.
- **Baselines.** The mean, persistence, and "always not poor" for classification.

## Limitations

- For first construction events, age counts from entry into LTPP rather than original construction.
- There are no layer thickness, material, precipitation or maintenance data.
- LTPP sections are more closely monitored than typical networks, so results may not transfer directly to other road networks.

## Future work

- Add true construction age, pavement structure and precipitation.
- Use mixed-effects models of section-specific deterioration.
- Try gradient boosting.
- Apply the framework to road condition data from Ghana and West Africa.

## Citation

If you use this work, please cite:

> Badu, D. (2026) *How much do traffic and climate explain pavement roughness? Regression and classification models of the International Roughness Index on LTPP data, with location-grouped validation.* GitHub repository.

## Licence

Code is released under the MIT Licence. LTPP data are provided by the US Federal Highway Administration.
