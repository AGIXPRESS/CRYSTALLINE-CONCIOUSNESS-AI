#!/usr/bin/env python3
"""
Script to fix the metal_ops.py file by removing the duplicate _mutuality_field_metal function
and ensuring that only our new implementation remains.
"""

import re

# Open the file
with open('Python/metal_ops.py', 'r') as f:
    content = f.read()

# Find the first occurrence of the function definition
first_func_pos = content.find("def _mutuality_field_metal(x, grid_size, interference_scale, decay_rate, dt):")
# Find the start of the next function
next_func_pos = content.find("def _mutuality_field_fallback", first_func_pos)

# Find the second occurrence of the function definition
second_func_pos = content.find("def _mutuality_field_metal(x, grid_size, interference_scale, decay_rate, dt):", next_func_pos)

# Find the end of the file or the next function definition after the second occurrence
file_end = len(content)
next_func_after_second = content.find("def ", second_func_pos + 1)
if next_func_after_second != -1:
    third_func_pos = next_func_after_second
else:
    third_func_pos = file_end

# Replace the first function definition with the second one
new_content = content[:first_func_pos] + content[second_func_pos:third_func_pos] + content[next_func_pos:]

# Save the fixed file
with open('Python/metal_ops.py', 'w') as f:
    f.write(new_content)

print("Fixed metal_ops.py by removing the duplicate function definition.")
