import pandas as pd

print("Reading dataset...")

# Read dataset
df = pd.read_csv("datasets/sf311.csv", low_memory=False)

# Keep only required columns
df = df[["Category", "Request Type", "Request Details"]]

# Map SF categories to your project categories
category_map = {
    "Street and Sidewalk Cleaning": "Garbage",
    "Litter Receptacles": "Garbage",
    "Street Defects": "Road Damage",
    "Sewer Issues": "Drainage Blockage",
    "Streetlights": "Streetlight Fault",
}

# Apply mapping
df["FinalCategory"] = df["Category"].map(category_map)

# Water leakage mapping using Request Type
water_types = [
    "Water_leak",
    "Water_Main_Break"
]

df.loc[df["Request Type"].isin(water_types), "FinalCategory"] = "Water Leakage"

# Keep only mapped categories
df = df[df["FinalCategory"].notna()]

# Keep only the required columns
df = df[["Request Details", "FinalCategory"]]

# Rename columns to match your project
df.columns = ["complaint_text", "category"]

# Remove empty rows
df = df.dropna()

print("\nOriginal Dataset Distribution:")
print(df["category"].value_counts())

# -------------------------------
# Balance the dataset
# -------------------------------

TARGET_PER_CLASS = 8000

balanced_parts = []

for category in sorted(df["category"].unique()):
    category_df = df[df["category"] == category]

    if len(category_df) >= TARGET_PER_CLASS:
        category_df = category_df.sample(
            n=TARGET_PER_CLASS,
            random_state=42
        )

    balanced_parts.append(category_df)

# Combine all categories
df = pd.concat(balanced_parts)

# Shuffle
df = df.sample(frac=1, random_state=42).reset_index(drop=True)

print("\nBalanced Dataset Distribution:")
print(df["category"].value_counts())

# Save
df.to_csv("data/complaints_text.csv", index=False)

print(f"\nSaved {len(df)} complaints to data/complaints_text.csv")
print("\nDataset preparation completed successfully!")