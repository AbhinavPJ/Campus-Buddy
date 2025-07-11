input_file = "passages_dump.txt"
output_file = "passages_cleaned.txt"

# Read the original content
with open(input_file, "r", encoding="utf-8") as f:
    text = f.read()

# Replace all occurrences
cleaned_text = text.replace("Link:", "Link to Course website")

# Save the modified content
with open(output_file, "w", encoding="utf-8") as f:
    f.write(cleaned_text)

print(f"✅ Replaced all occurrences and saved to {output_file}")
