import os
import re

# Paths
input_file = "hyperlinks.txt"
pyq_base_path = "links/all_courses_pyq"
output_file = "course_info.txt"

# Prefixes to exclude from pyq links
excluded_prefixes = [
    "https://bsw.iitd.ac.in/forms/",
    "https://bsw.iitd.ac.in/misc/"
]

# Parse hyperlinks.txt
courses = []
with open(input_file, 'r') as f:
    lines = [line.strip() for line in f if line.strip()]
    i = 0
    while i < len(lines):
        name_match = re.match(r"Name:\s*(.+)", lines[i])
        link_match = re.match(r"Link:\s*(.+)", lines[i+1]) if i+1 < len(lines) and lines[i+1].startswith("Link:") else None
        name = name_match.group(1) if name_match else None
        link = link_match.group(1) if link_match else "N/A"
        courses.append((name, link))
        i += 2 if link_match else 1

# Write to course_info.txt
with open(output_file, 'w') as f:
    for name, link in courses:
        f.write(f"Name: {name}\n")
        f.write(f"Course: {name}\n")
        f.write(f"Link: {link}\n")
        pyq_path = os.path.join(pyq_base_path, f"{name}.txt")
        if os.path.exists(pyq_path):
            with open(pyq_path, 'r') as pyq_file:
                pyqs = [line.strip() for line in pyq_file if line.strip()]
                for pyq in pyqs:
                    if not any(pyq.startswith(prefix) for prefix in excluded_prefixes):
                        f.write(f"pyq: {pyq}\n")
        f.write("\n")  # Separate entries

print("✅ Filtered course data with valid PYQ links saved to course_info.txt")
