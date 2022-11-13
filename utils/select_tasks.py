'''Copy selected tasks (by #) from natural-instructions-master to selected_tasks'''

import os
import re
import shutil


pattern = r"task(\d+)\_"
files = []
lower = 937
upper = 1388
for f in os.listdir('natural-instructions-master/tasks'):
    if f.endswith('.json'):
        task_num = int(re.findall(pattern, f)[0])
        if task_num <= upper and task_num >= lower:
            files.append(f)

src = 'natural-instructions-master/tasks/'
dst = 'selected_tasks/'
for f in files:
    shutil.copyfile(src+f, dst+f)