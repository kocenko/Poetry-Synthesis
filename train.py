file_path = "data/poe_data.txt"
with open(file_path, encoding="utf-8") as f:
    text: str = f.read()

print(f"Number of characters in the dataset: {len(text)}")