from metaflow import FlowSpec, step, Parameter
import pandas as pd
import mlflow.pyfunc
import os

class ScoringFlow(FlowSpec):

    data_path = Parameter("data-path", help="Path to new data CSV to score")

    @step
    def start(self):
        print("Scoring flow started")
        self.next(self.load_data)

    @step
    def load_data(self):
        self.new_data = pd.read_csv(self.data_path)
        self.next(self.preprocess_data)

    @step
    def preprocess_data(self):
        pcos = self.new_data.copy()

        # -- DATA PREPROCESSING -------------------------------------------------------
        pcos = pcos.set_index("ID")
        pcos = pcos.iloc[:,[1,0] + list(range(2,len(pcos.columns)))]

        # -- FIXING CATEGORY MISMATCHES -----------------------------------------------
        # Age
        pcos.loc[(pcos["Age"] == "Less than 20") | (pcos["Age"] == "Less than 20-25"), "Age"] = "15-20"

        # Hormonal_Imbalance
        pcos.loc[pcos["Hormonal_Imbalance"] == "Yes Significantly", "Hormonal_Imbalance"] = "Yes"
        pcos.loc[pcos["Hormonal_Imbalance"] == "No, Yes, not diagnosed by a doctor", "Hormonal_Imbalance"] = "Unknown"

        # Hirsutism
        pcos.loc[pcos["Hirsutism"] == "No, Yes, not diagnosed by a doctor", "Hirsutism"] = "Unknown"

        # Conception_Difficulty
        pcos.loc[pcos["Conception_Difficulty"] == "Yes, diagnosed by a doctor", "Conception_Difficulty"] = "Yes"
        pcos.loc[pcos["Conception_Difficulty"] == "No, Yes, not diagnosed by a doctor", "Conception_Difficulty"] = "Unknown"

        # Insulin_Resistance
        pcos.loc[pcos["Insulin_Resistance"] == "No, Yes, not diagnosed by a doctor", "Insulin_Resistance"] = "Unknown"

        # Exercise_Frequency
        pcos.loc[pcos["Exercise_Frequency"].isin(["6-8 hours", "Less than 6 hours", "Less than usual"]), "Exercise_Frequency"] = "1-2 Times a Week"

        # Exercise_Type
        pcos["Exercise_Type"] = pcos["Exercise_Type"].fillna("Unknown")
        pcos.loc[pcos["Exercise_Type"].str.startswith("Cardio"), "Exercise_Type"] = "Cardio"
        pcos.loc[pcos["Exercise_Type"].str.startswith("Flexibility"), "Exercise_Type"] = "Flexibility and balance"
        pcos.loc[pcos["Exercise_Type"].str.startswith("Strength"), "Exercise_Type"] = "Strength training"
        pcos.loc[pcos["Exercise_Type"] == "Somewhat", "Exercise_Type"] = "Unknown"

        # Exercise_Duration
        pcos.loc[pcos["Exercise_Duration"] == "20 minutes", "Exercise_Duration"] = "Less than 30 minutes"
        pcos.loc[pcos["Exercise_Duration"] == "30 minutes to 1 hour", "Exercise_Duration"] = "More than 30 minutes"
        pcos.loc[pcos["Exercise_Duration"] == "45 minutes", "Exercise_Duration"] = "More than 30 minutes"
        pcos.loc[pcos["Exercise_Duration"] == "Less than 6 hours", "Exercise_Duration"] = "Not Applicable"

        # Sleep Hours
        pcos.loc[pcos["Sleep_Hours"] == "3-4 hours", "Sleep_Hours"] = "Less than 6 hours"

        # Target Variable: PCOS
        # pcos.loc[pcos["PCOS"] == "Yes", "PCOS"] = 1
        # pcos.loc[pcos["PCOS"] == "No", "PCOS"] = 0

        # -- MISSING VALUE INPUTATION -------------------------------------------------
        # Mean for the Weight Column
        mean_weight = float(pcos["Weight_kg"].mean())
        pcos["Weight_kg"] = pcos["Weight_kg"].fillna(mean_weight)

        for col in pcos.iloc[:, 1:].columns:
            mode_value = pcos[col].mode()[0]
            pcos[col] = pcos[col].fillna(mode_value)

        # -- REORDERING COLUMNS -------------------------------------------------------
        pcos = pcos.iloc[:, [0, 1] + list(range(3,len(pcos.columns))) + [2]]

        # -- CONVERTING TO CATEGORY ---------------------------------------------------
        for col in pcos.iloc[:, 1:12].columns:
            pcos[col] = pcos[col].astype('category')

        self.transformed_data = pcos
        self.next(self.load_model)

    @step
    def load_model(self):
        model_uri = "models:/PCOS_Model/1"  # You can adjust version or use "latest"
        self.model = mlflow.pyfunc.load_model(model_uri)
        self.next(self.predict)

    @step
    def predict(self):
        df = self.transformed_data
        self.predictions = self.model.predict(df)
        self.result_df = df.copy()
        self.result_df["Prediction"] = self.predictions
        self.next(self.end)

    @step
    def end(self):
        output_path = "scored_predictions.csv"
        self.result_df.to_csv(output_path, index=False)
        print(f"Predictions saved to {output_path}")

if __name__ == '__main__':
    ScoringFlow()