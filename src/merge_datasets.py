import pandas as pd

print("Loading datasets...")

real = pd.read_csv("data/complaints_text.csv")
manual = pd.read_csv("data/manual_complaints_5000.csv")

print(f"Real dataset: {len(real)}")
print(f"Manual dataset: {len(manual)}")

# Merge WITHOUT removing duplicates
combined = pd.concat([real, manual], ignore_index=True)

# Shuffle the rows
combined = combined.sample(frac=1, random_state=42).reset_index(drop=True)

print("\nFinal Distribution:")
print(combined["category"].value_counts())

print(f"\nTotal rows: {len(combined)}")

# Save as a new file
combined.to_csv("data/complaints_text_merged.csv", index=False)

print("\nSaved to data/complaints_text_merged.csv")