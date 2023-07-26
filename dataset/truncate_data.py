import json
import sys
import os.path
from os import listdir
from os.path import isfile, join
from concurrent import futures
import re

class TruncateFiles:
    def __init__(self, category, batch_size, body_size, num_workers, origin_path, target_path):
        self.pool = futures.ThreadPoolExecutor(max_workers=num_workers)    
        self.origin_path = origin_path
        self.target_path = target_path
        self.category = category
        self.batch_size = batch_size
        self.body_size = body_size
        self.paths = [f for f in listdir(self.origin_path + self.category + "/") if isfile(join(self.origin_path + self.category + "/", f))]
        
    def clean_text(self, text):
        text = text.lower()
        text = re.sub('\S*@\S*\s?', '', text)
        text = re.sub('https?://[A-Za-z0-9]','',text)
        text = re.sub("[^a-z' ]+", ' ', text)
        text = ' ' + text + ' '
        text = re.sub("[^a-z]'|'[^a-z]", ' ', text)
        return text

    def truncate_files(self, filepaths):
        for filepath in filepaths:
            file = self.load_file(filepath)
            file["title"] = self.clean_text(file["title"])
            file["body"] = self.clean_text(file["body"])[:self.body_size]
            self.write_file(filepath, file)

    def load_file(self, filepath):
        with open(join(self.origin_path, self.category, filepath), 'r') as f:
            return json.load(f)

    def write_file(self, filepath, file):
        with open(join(self.target_path, self.category, filepath), 'w') as f1:
            return json.dump(file, f1)
            
    def run(self):
        jobs = list()
        for i in range(0, len(self.paths), self.batch_size):
            try:
                filepaths = self.paths[i:(i + self.batch_size)]
                job = self.pool.submit(
                    self.truncate_files, filepaths
                )
                jobs.append(job)
            except Exception as e:
                print(e)

def main():
    truncate = TruncateFiles(category=str(sys.argv[1]), batch_size=30, body_size=1024, num_workers=25, origin_path=str(sys.argv[2]), target_path=str(sys.argv[3]))
    truncate.run()

if __name__ == "__main__":
    main()
            

