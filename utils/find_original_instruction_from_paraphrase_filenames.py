input_file = "splits/test_tasks_v2.txt"
output_file = "splits/test_tasks_v2_original.txt"

with open(input_file) as file:
    lines = [line.rstrip() for line in file]
    original_lines = []
    for line in lines:
        line = line.split("_gpt3")[0]
        original_lines.append(line)

original_lines = list(set(original_lines))

with open(output_file, 'w') as f:
    for line in original_lines:
        f.write(f"{line}\n")