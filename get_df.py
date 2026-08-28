import os
import pandas as pd

base_path = r"E:\interview preparation guide\leetcode-companywise-interview-questions"

all_data = []

for company in os.listdir(base_path):
    company_path = os.path.join(base_path, company)

    csv_path = os.path.join(company_path, "all.csv")

    if os.path.exists(csv_path):
        df = pd.read_csv(csv_path)
        df["company"] = company
        all_data.append(df)

final_df = pd.concat(all_data, ignore_index=True)
print(final_df.shape)
print(final_df.head())
import os

os.makedirs("data", exist_ok=True)

final_df.to_csv("data/final_dataset.csv", index=False)