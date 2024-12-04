import os
import re
from collections import defaultdict


def rename_images(directory):
    # Dictionary to count occurrences of each combination (e.g., '5L')
    count_dict = defaultdict(int)

    # Collect all image files in the directory
    image_files = [f for f in os.listdir(directory) if re.match(r'.*\d[L|R]\.\w+$', f)]

    # Dictionary to store the new names for renaming
    new_names = {}

    # First pass: count the occurrences of each pattern
    for image_file in image_files:
        pattern = image_file[-6:-4]
        count_dict[pattern] += 1
        new_name = f"{pattern}{count_dict[pattern]}{image_file[-4:]}"
        new_names[image_file] = new_name

    # Second pass: rename the files
    for old_name, new_name in new_names.items():
        old_path = os.path.join(directory, old_name)
        new_path = os.path.join(directory, new_name)
        os.rename(old_path, new_path)
        print(f"Renamed {old_name} to {new_name}")


# Example usage
directory_path = "D:\\Android development\\archive (4)\\test"
rename_images(directory_path)
