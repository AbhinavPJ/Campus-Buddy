import torch

# Load the file
data = torch.load("hybrid_embeddings.pt")

# Get passages (list of strings)
passages = data["passages"]

# Output file path
output_file = "passages_dump.txt"

# Write each passage to the file, separated by two newlines
with open(output_file, "w", encoding="utf-8") as f:
    for i, passage in enumerate(passages):
        f.write(passage.strip() + "\n\n")

print(f"✅ Saved {len(passages)} passages to {output_file}")
