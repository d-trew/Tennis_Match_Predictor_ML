# Tennis Match Outcome Predictor

A Python machine learning project that predicts professional tennis match outcomes from historical match data, with a particular focus on **preventing data leakage**.

### Tech Stack

* **Python**
* **Pandas / NumPy**
* **Scikit-learn**
* **XGBoost**
* **Optuna**

### What I Did

* Processed **74,000+ historical matches** and generated player-perspective training examples.
* Engineered features including ranking, form, surface performance, head-to-head and fatigue.
* Built Random Forest and XGBoost models and experimented with stacking.
* Used Optuna for hyperparameter optimisation.
* Implemented chronological player-history calculations so features only used information available **before each match**.
* Investigated temporal and player-exclusive validation to identify potential leakage and overly optimistic results.

### Key Takeaway

An end-to-end ML project covering **data preparation → feature engineering → modelling → evaluation**, with particular emphasis on whether the predictions can actually be trusted.


