import wget

tasks_txt = "./train_tasks.txt"
output_directory = "./tk-instruct-train-tasks/"
url_base = "https://raw.githubusercontent.com/allenai/natural-instructions/master/tasks/"

def map_to_url(filename):
    return url_base + filename + ".json"

with open(tasks_txt) as f:
    filenames = [filename.rstrip() for filename in f]

for filename in filenames:
    url = map_to_url(filename)
    try:
        wget.download(url, output_directory)
    except:
        print(f"{url} not downloaded")