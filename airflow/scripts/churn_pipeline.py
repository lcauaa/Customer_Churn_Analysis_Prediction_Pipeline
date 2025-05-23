# import os
# from dotenv import load_dotenv
# import boto3
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score
from sklearn.model_selection import StratifiedKFold, cross_val_score, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, RobustScaler


class CostumerChurnPredictor:
    """
    CUSTOMER PRDICTION CLASS
    """

    def __init__(self):
        self.models = {}
        self.feature_importance = None
        self.scaler = RobustScaler()

    def feature_engineering(self, df):
        """
        FEATURE ENGINEERING
        """

        df = df.copy()

        df["TotalCharges"] = pd.to_numeric(df["TotalCharges"], errors="coerce")
        df["TotalCharges"] = df["TotalCharges"].fillna(df["TotalCharges"].median())

        df["AvgMonthlyCharges"] = df["TotalCharges"] / (df["tenure"] + 1)
        df["CustomerValue"] = df["TotalCharges"] * (1 + df["tenure"] / 100)

        service_cols = [
            "PhoneService",
            "MultipleLines",
            "OnlineSecurity",
            "OnlineBackup",
            "DeviceProtection",
            "TechSupport",
            "StreamingTV",
            "StreamingMovies",
        ]
        df["ServiceComplexity"] = 0
        for col in service_cols:
            df["ServiceComplexity"] += (df[col] == "Yes").astype(int)

        df["HighRisk_MonthToMonth"] = (df["Contract"] == "Month-to-month").astype(int)
        df["HighRisk_ElectronicCheck"] = (
            df["PaymentMethod"] == "Electronic check"
        ).astype(int)
        df["HighRisk_PaperlessBilling"] = (df["PaperlessBilling"] == "Yes").astype(int)

        def categorize_tenure(tenure):
            if tenure <= 6:
                return "New"
            elif tenure <= 24:
                return "Growing"
            elif tenure <= 48:
                return "Mature"
            else:
                return "Veteran"

        df["LifecycleStage"] = df["tenure"].apply(categorize_tenure)

        df["PricePerService"] = df["MonthlyCharges"] / (df["ServiceComplexity"] + 1)
        df["TotalChargesPerYear"] = df["TotalCharges"] / ((df["tenure"] / 12) + 0.1)

        contract_scores = {"Month-to-month": 1, "One year": 2, "Two year": 3}
        df["ContractLoyaltyScore"] = df["Contract"].map(contract_scores)

        return df

    def prepare_data(self, df):
        """
        DATA PREPROCESSING PIPELINE
        """

        df_engineered = self.feature_engineering(df)

        X = df_engineered.drop(["customerID", "Churn"], axis=1)
        y = df_engineered["Churn"].map({"Yes": 1, "No": 0})

        categorical_cols = X.select_dtypes(include=["object"]).columns.tolist()
        numerical_cols = X.select_dtypes(include=["int64", "float64"]).columns.tolist()

        numerical_transformer = Pipeline(steps=[("scaler", RobustScaler())])

        categorical_transformer = Pipeline(
            steps=[
                (
                    "onehot",
                    OneHotEncoder(
                        drop="first", sparse_output=False, handle_unknown="ignore"
                    ),
                )
            ]
        )

        preprocessor = ColumnTransformer(
            transformers=[
                ("num", numerical_transformer, numerical_cols),
                ("cat", categorical_transformer, categorical_cols),
            ]
        )

        return X, y, preprocessor

    def build_ensemble_models(self):
        """
        BUILD MODELS, MAYBE ADD XGBOOST OR SOME OTHERS BOOSTS LATER
        """

        models = {
            "RandomForest": RandomForestClassifier(
                n_estimators=200,
                max_depth=10,
                min_samples_split=5,
                min_samples_leaf=2,
                random_state=42,
            ),
            "LogisticRegression": LogisticRegression(
                C=0.1, penalty="l2", random_state=42, max_iter=1000
            ),
        }

        return models

    def train_and_evaluate(self, df):
        """
        COMPLETE TRAINING AND EVALUATION PIPELINE
        """
        print("Starting Churn Prediction Pipeline...")

        X, y, preprocessor = self.prepare_data(df)

        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42, stratify=y
        )

        print(f"Training set: {X_train.shape[0]} samples")
        print(f"Test set: {X_test.shape[0]} samples")
        print(f"Churn rate: {y.mean():.2%}")

        models = self.build_ensemble_models()

        results = {}
        trained_models = {}

        for name, model in models.items():
            print(f"Training {name}...")

            pipeline = Pipeline([("preprocessor", preprocessor), ("classifier", model)])

            cv_scores = cross_val_score(
                pipeline,
                X_train,
                y_train,
                cv=StratifiedKFold(n_splits=5, shuffle=True, random_state=42),
                scoring="roc_auc",
            )

            pipeline.fit(X_train, y_train)

            # y_pred = pipeline.predict(X_test)
            y_pred_proba = pipeline.predict_proba(X_test)[:, 1]

            auc_score = roc_auc_score(y_test, y_pred_proba)

            results[name] = {
                "CV_AUC_Mean": cv_scores.mean(),
                "CV_AUC_Std": cv_scores.std(),
                "Test_AUC": auc_score,
                "Pipeline": pipeline,
            }

            trained_models[name] = pipeline

            print(f"{name} - CV AUC: {cv_scores.mean():.4f} (±{cv_scores.std():.4f})")
            print(f"{name} - Test AUC: {auc_score:.4f}")

        best_model_name = max(results, key=lambda x: results[x]["Test_AUC"])
        self.best_model = trained_models[best_model_name]

        print(f"BEST MODEL: {best_model_name}")
        print(f"Best Test AUC: {results[best_model_name]['Test_AUC']:.4f}")

        self.extract_feature_importance(X.columns)

        self.final_evaluation(X_test, y_test)

        return results

    def extract_feature_importance(self, feature_names):
        """
        EXTRACT FEATURE IMPORTANCE FOR BUSINESS INSIGHTS
        """

        try:
            preprocessor = self.best_model.named_steps["preprocessor"]
            classifier = self.best_model.named_steps["classifier"]

            if hasattr(classifier, "feature_importances_"):
                importances = classifier.feature_importances_

                cat_features = (
                    preprocessor.named_transformers_["cat"]
                    .named_steps["onehot"]
                    .get_feature_names_out()
                )
                num_features = preprocessor.named_transformers_[
                    "num"
                ].get_feature_names_out()
                all_features = list(num_features) + list(cat_features)

                importance_df = pd.DataFrame(
                    {"Feature": all_features, "Importance": importances}
                ).sort_values("Importance", ascending=False)

                self.feature_importance = importance_df

                print("MOST IMPORTANT FEATURES:")
                print(importance_df.head(10).to_string(index=False))

        except Exception as e:
            print(f"Could not extract feature importance: {e}")

    def final_evaluation(self, X_test, y_test):
        """
        COMPREHENSIVE FINAL EVALUATION WITH CSV EXPORT
        """

        y_pred = self.best_model.predict(X_test)
        y_pred_proba = self.best_model.predict_proba(X_test)[:, 1]

        print("FINAL MODEL PERFORMANCE REPORT\n")

        print(f"ROC AUC Score: {roc_auc_score(y_test, y_pred_proba):.4f}")

        print("CLASSIFICATION REPORT:")
        print(classification_report(y_test, y_pred))

        print("CONFUSION MATRIX\n:")
        cm = confusion_matrix(y_test, y_pred)
        print(cm)

        true_positives = cm[1, 1]

        retention_cost_per_customer = 100
        lost_revenue_per_churner = 1000

        cost_saved = true_positives * lost_revenue_per_churner
        retention_cost = true_positives * retention_cost_per_customer
        net_benefit = cost_saved - retention_cost

        print("\nBUSINESS IMPACT ANALYSIS:")
        print(f"Customers at risk identified: {true_positives}")
        print(f"Potential revenue saved: ${cost_saved:,}")
        print(f"Retention campaign cost: ${retention_cost:,}")
        print(f"Net business benefit: ${net_benefit:,}")

        self.export_predictions_to_csv(X_test, y_test, y_pred, y_pred_proba)

    def export_predictions_to_csv(self, X_test, y_test, y_pred, y_pred_proba):
        """
        EXPORT DETAILED PREDICTIONS TO CSV
        """
        try:
            predictions_df = pd.DataFrame(
                {
                    "Actual_Churn": y_test.values,
                    "Predicted_Churn": y_pred,
                    "Churn_Probability": y_pred_proba,
                    "Risk_Category": pd.cut(
                        y_pred_proba,
                        bins=[0, 0.3, 0.7, 1.0],
                        labels=["Low_Risk", "Medium_Risk", "High_Risk"],
                    ),
                    "Prediction_Confidence": np.where(
                        y_pred_proba > 0.5, y_pred_proba, 1 - y_pred_proba
                    ),
                }
            )

            predictions_df["Customer_Index"] = X_test.index

            predictions_df = predictions_df.sort_values(
                "Churn_Probability", ascending=False
            )

            predictions_df.to_csv(
                "/opt/airflow/data/customer_churn_predictions.csv", index=False
            )
            print("Predictions exported to 'customer_churn_predictions.csv'")
            print(f"Total customers analyzed: {len(predictions_df)}")
            print(
                f"High-risk customers: {sum(predictions_df['Risk_Category'] == 'High_Risk')}"
            )

        except Exception as e:
            print(f"Error exporting predictions: {e}")

    def predict_new_customers(self, new_customer_data, export_csv=True):
        """
        PREDICT CHURN FOR NEW CUSTOMERS AND EXPORT RESULTS
        """

        try:
            new_data_engineered = self.feature_engineering(new_customer_data)

            X_new = new_data_engineered.drop(["customerID"], axis=1)
            if "Churn" in X_new.columns:
                X_new = X_new.drop(["Churn"], axis=1)

            predictions = self.best_model.predict(X_new)
            probabilities = self.best_model.predict_proba(X_new)[:, 1]

            results_df = pd.DataFrame(
                {
                    "CustomerID": new_customer_data["customerID"],
                    "Churn_Probability": probabilities,
                    "Predicted_Churn": predictions,
                    "Risk_Level": pd.cut(
                        probabilities,
                        bins=[0, 0.3, 0.7, 1.0],
                        labels=["Low_Risk", "Medium_Risk", "High_Risk"],
                    ),
                    "Recommended_Action": probabilities.apply(
                        lambda x: "Immediate_Retention_Campaign"
                        if x > 0.7
                        else "Monitor_Closely"
                        if x > 0.3
                        else "Standard_Service"
                    ),
                }
            )

            results_df = results_df.sort_values("Churn_Probability", ascending=False)

            if export_csv:
                results_df.to_csv(
                    "/opt/airflow/data/new_customer_churn_predictions.csv", index=False
                )
                print(
                    "New customer predictions exported to 'new_customer_churn_predictions.csv'"
                )

            return results_df

        except Exception as e:
            print(f"Error predicting new customers: {e}")
            return None


def run_churn_prediction():
    """
    EXECUTE THE CHURN PREDICTION PIPELINE
    """

    df = pd.read_csv("/opt/airflow/data/tableau_ready_customer_churn.csv")

    predictor = CostumerChurnPredictor()
    results = predictor.train_and_evaluate(df)

    return results


if __name__ == "__main__":
    predictor, results = run_churn_prediction()

    print("\nPredictions file was created :)")
