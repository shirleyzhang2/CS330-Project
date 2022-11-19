import os
import shutil

src_dir = 'tk-instruct-train-tasks'
dst_dir = 'tk-instruct-train-classfication-tasks'

if not os.path.exists(dst_dir):
    os.mkdir(dst_dir)
    
for file in os.listdir(src_dir):
    if 'classification' not in file:
        continue
    src = os.path.join(src_dir, file)
    dst = os.path.join(dst_dir, file)
    shutil.copy(src, dst)