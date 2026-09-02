import glob
import os
import warnings

import pandas as pd

warnings.filterwarnings("ignore")


class TrainDataGenerator:
    """Generate trajectory-level training data from AuctionNet CSV files."""

    def __init__(self, file_folder_path="./data/traffic"):
        self.file_folder_path = file_folder_path
        self.training_data_path = os.path.join(
            self.file_folder_path, "training_data_rlData_folder"
        )

    def batch_generate_train_data(self):
        os.makedirs(self.training_data_path, exist_ok=True)
        csv_files = glob.glob(os.path.join(self.file_folder_path, "*.csv"))
        print(csv_files)
        training_data_list = []
        for csv_path in csv_files:
            print("Processing file:", csv_path)
            df = pd.read_csv(csv_path)
            df_processed = self._generate_train_data(df)
            csv_filename = os.path.basename(csv_path)
            train_data_filename = csv_filename.replace(".csv", "-rlData.csv")
            train_data_path = os.path.join(
                self.training_data_path, train_data_filename
            )
            df_processed.to_csv(train_data_path, index=False)
            training_data_list.append(df_processed)
            del df, df_processed
            print("Processed file:", csv_path)

        combined_dataframe = pd.concat(
            training_data_list, axis=0, ignore_index=True
        )
        combined_dataframe_path = os.path.join(
            self.training_data_path, "training_data_all-rlData.csv"
        )
        combined_dataframe.to_csv(combined_dataframe_path, index=False)
        print("Combined training data saved to:", combined_dataframe_path)

    def _generate_train_data(self, df):
        """Construct reinforcement-learning trajectories from raw auction data."""
        training_data_rows = []

        group_columns = [
            "deliveryPeriodIndex",
            "advertiserNumber",
            "advertiserCategoryIndex",
            "budget",
            "CPAConstraint",
        ]
        for (
            delivery_period_index,
            advertiser_number,
            advertiser_category_index,
            budget,
            cpa_constraint,
        ), group in df.groupby(group_columns):
            group = group.sort_values("timeStepIndex")

            group["timeStepIndex_volume"] = group.groupby(
                "timeStepIndex"
            )["timeStepIndex"].transform("size")
            timestep_volume_sum = group.groupby("timeStepIndex")[
                "timeStepIndex_volume"
            ].first()

            historical_volume = (
                timestep_volume_sum.cumsum().shift(1).fillna(0).astype(int)
            )
            group["historical_volume"] = group["timeStepIndex"].map(
                historical_volume
            )

            last_three_volume = (
                timestep_volume_sum.rolling(window=3, min_periods=1)
                .sum()
                .shift(1)
                .fillna(0)
                .astype(int)
            )
            group["last_3_timeStepIndexs_volume"] = group[
                "timeStepIndex"
            ].map(last_three_volume)

            group_agg = (
                group.groupby("timeStepIndex")
                .agg(
                    {
                        "bid": "mean",
                        "leastWinningCost": "mean",
                        "conversionAction": "mean",
                        "xi": "mean",
                        "pValue": "mean",
                        "timeStepIndex_volume": "first",
                    }
                )
                .reset_index()
            )

            for column in [
                "bid",
                "leastWinningCost",
                "conversionAction",
                "xi",
                "pValue",
            ]:
                group_agg[f"avg_{column}_all"] = (
                    group_agg[column].expanding().mean().shift(1)
                )
                group_agg[f"avg_{column}_last_3"] = (
                    group_agg[column]
                    .rolling(window=3, min_periods=1)
                    .mean()
                    .shift(1)
                )

            group = group.merge(
                group_agg, on="timeStepIndex", suffixes=("", "_agg")
            )
            real_all_cost = (group["isExposed"] * group["cost"]).sum()
            real_all_conversion = group["conversionAction"].sum()

            for timestep_index in group["timeStepIndex"].unique():
                timestep_data = group[group["timeStepIndex"] == timestep_index]
                timestep_data.fillna(0, inplace=True)

                num_timesteps = 48
                budget = timestep_data["budget"].iloc[0]
                remaining_budget = timestep_data["remainingBudget"].iloc[0]
                time_left = (num_timesteps - timestep_index) / num_timesteps
                budget_left = remaining_budget / budget if budget > 0 else 0
                features = timestep_data.iloc[0].to_dict()

                state = (
                    time_left,
                    budget_left,
                    features["avg_bid_all"],
                    features["avg_bid_last_3"],
                    features["avg_leastWinningCost_all"],
                    features["avg_pValue_all"],
                    features["avg_conversionAction_all"],
                    features["avg_xi_all"],
                    features["avg_leastWinningCost_last_3"],
                    features["avg_pValue_last_3"],
                    features["avg_conversionAction_last_3"],
                    features["avg_xi_last_3"],
                    features["pValue_agg"],
                    features["timeStepIndex_volume_agg"],
                    features["last_3_timeStepIndexs_volume"],
                    features["historical_volume"],
                )

                total_bid = timestep_data["bid"].sum()
                total_value = timestep_data["pValue"].sum()
                action = total_bid / total_value if total_value > 0 else 0
                exposed = timestep_data[timestep_data["isExposed"] == 1]
                reward = exposed["conversionAction"].sum()
                reward_continuous = exposed["pValue"].sum()
                done = int(
                    timestep_index == num_timesteps - 1
                    or timestep_data["isEnd"].iloc[0] == 1
                )

                training_data_rows.append(
                    {
                        "deliveryPeriodIndex": delivery_period_index,
                        "advertiserNumber": advertiser_number,
                        "advertiserCategoryIndex": advertiser_category_index,
                        "budget": budget,
                        "CPAConstraint": cpa_constraint,
                        "realAllCost": real_all_cost,
                        "realAllConversion": real_all_conversion,
                        "timeStepIndex": timestep_index,
                        "state": state,
                        "action": action,
                        "reward": reward,
                        "reward_continuous": reward_continuous,
                        "done": done,
                    }
                )

        training_data = pd.DataFrame(training_data_rows)
        training_data = training_data.sort_values(
            by=[
                "deliveryPeriodIndex",
                "advertiserNumber",
                "timeStepIndex",
            ]
        )
        trajectory_columns = ["deliveryPeriodIndex", "advertiserNumber"]
        training_data["next_state"] = training_data.groupby(
            trajectory_columns
        )["state"].shift(-1)
        training_data.loc[training_data["done"] == 1, "next_state"] = None
        return training_data


def run_generate_train_data():
    data_generator = TrainDataGenerator(file_folder_path="./data/traffic")
    data_generator.batch_generate_train_data()


if __name__ == "__main__":
    run_generate_train_data()
