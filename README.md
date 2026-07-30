# Diabetes-Prediction-Model
Diabetes Prediction Using Machine Learning
Domain
This project belongs to the domain of Healthcare Informatics and Applied Machine Learning, focusing on predictive modeling for non‑communicable diseases (NCDs). Among the major NCDs—cardiovascular disease, cancer, chronic respiratory disease, and diabetes—this study specifically addresses diabetes mellitus, a condition characterized by elevated blood glucose and impaired metabolism of fats and proteins. Diabetes has become a major global health challenge, with prevalence rising rapidly in low‑ and middle‑income countries (World Health Organization, 2016). The disease is strongly linked to cardiovascular complications, kidney failure, and premature mortality, making early detection critical for public health (Susan van Dijk, Beulens, van der Schouw, Grobbee, & Neal, 2010).

Problem Statement
Diabetes is a leading cause of premature death worldwide. Previous work has used the Pima Indians Diabetes Dataset (Smith et al., 1988) as a benchmark, achieving ~70–80% accuracy with models like Logistic Regression and SVM (Han et al., 2020). However, Pima is limited in size and scope. This project uses the Kaggle Diabetes Prediction Dataset (Mustafa, 2021) with ~100,000 records from South Asian hospitals, providing richer features and greater relevance for Pakistan.

Dataset link: https://www.kaggle.com/datasets/iammustafatz/diabetes-prediction-dataset

Dataset Features
Age → Risk increases with age.
Gender → Biological differences influence susceptibility.
BMI → Obesity is the strongest risk factor.
Hypertension → Common comorbidity with diabetes.
Heart Disease → Cardiovascular complication.
Smoking History → Lifestyle factor affecting risk.
HbA1c Level → Long‑term blood sugar indicator.
Blood Glucose Level → Immediate measure of blood sugar.
Diabetes (0/1) → Target variable.
Machine Learning Models
This is a classical ML problem (tabular data), so deep learning is not required. Models used:

Logistic Regression → Baseline, interpretable.
Decision Tree → Simple, visual.
Random Forest → Ensemble, high accuracy.
SVM → Strong classification boundaries.
KNN → Distance‑based, lifestyle features.
Gradient Boosting (XGBoost, CatBoost) → Advanced ensemble, high accuracy.
Expected Outcome
The deliverable is a validated predictive model achieving sensitivity ≥ 85%. The study will also identify which features contribute most to diabetes risk, providing clinicians with a prioritized screening checklist. SHAP values will be used for interpretability.
