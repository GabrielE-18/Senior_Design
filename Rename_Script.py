import os

# Path to your YOLO labels folder
labels_dir = "C:\\Users\\gabri\\School\\Senior\\Dataset\\val\\labels"

# Map all cow classes (0-9) to class 0, except for class 4 which will be set to class 1
for filename in os.listdir(labels_dir):
    if filename.endswith(".txt"):
        file_path = os.path.join(labels_dir, filename)

        with open(file_path, "r") as file:
            lines = file.readlines()

        new_lines = []
        for line in lines:
            parts = line.split()
            class_id = parts[0]

            # Set class ID to 0, except for class 4 which will be set to 1
            if class_id.isdigit():
                if int(class_id) == 4:
                    new_lines.append(f"1 {' '.join(parts[1:])}\n")  # Class 4 to 1
                else:
                    new_lines.append(f"0 {' '.join(parts[1:])}\n")  # All other classes to 0
            else:
                new_lines.append(line)

        # Save the updated labels
        with open(file_path, "w") as file:
            file.writelines(new_lines)

print("✅ Labels updated: All classes except 4 are set to 0, class 4 is set to 1.")
