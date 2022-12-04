import argparse
import os
import shutil
import json
import numpy as np
import re
from collections import Counter
from matplotlib import pyplot as plt

parser = argparse.ArgumentParser()
parser.add_argument('-i', "--input", help="input task dir containing task files")
args = parser.parse_args()
files = os.listdir(args.input)
regex = r"task[0-9]+_(.+).json"

#files = [re.findall(regex, file)[0] for file in files]
files = [cat for file in files for cat in json.load(open(os.path.join(args.input, file)))['Categories']]
count = Counter(files).most_common()

x = [c[0] for c in count]
y = [c[1] for c in count]
print(len(x))
fig, ax = plt.subplots()
plot = ax.bar(x, y, color='cornflowerblue', edgecolor='mediumblue')
ax.set_xticks(list(range(len(x))), x, rotation=90)
# ax.bar_label(plot) # displays the value
ax.set_ylabel('# of tasks of a type')
ax.set_title('Training task counts')
plt.legend()
plt.tight_layout()
plt.show()
#print(files[0], re.findall(regex, files[0])[0])