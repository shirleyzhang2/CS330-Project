import wget

tasks_txt = "./eval/textual_entailment_first10.txt"
output_directory = "./tasks/"
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