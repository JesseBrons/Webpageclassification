import hashlib
import click
import time
import glob
import gzip
import sys
import csv
import re
import os
import numpy as np
import pandas as pd
from os import path
from bs4 import BeautifulSoup
from concurrent import futures
from html.parser import HTMLParser
from urllib import error, parse, request
from socket import timeout as SocketTimeoutError
from http.client import IncompleteRead, InvalidURL
from resiliparse.extract.html2text import extract_plain_text

class GetWebsite:
    def __init__(self, target_folder):
        self.error_types = {"body_error": 1, "language_error": 2, "title_error": 3, "connection_error": 4, "timeout_error": 5, "redirects_error": 7, "general_error": 8, "url_error": 9, "path_error": 0}
        self.pool = futures.ThreadPoolExecutor(max_workers=25)
        self.data_folder = target_folder

    def load_url(self, website, timeout):
        url_hash = hashlib.sha256(website.url.encode('utf-8')).hexdigest()

        file_location = self.data_folder + website.category + "/" + url_hash + ".txt"

        if not path.exists(file_location):
            r = request.Request(
                url=website.url,
                headers={
                    "User-Agent": "Mozilla/5.0 (X11; Ubuntu; Linux x86_64; rv:72.0) Gecko/20100101 Firefox/72.0"
                },
            )

            result = {
                "category": website.category,
                "title": "",
                "body": "",
                "url": website.url,
                "keywords": ""
            }
        
            try:
                http_response = request.urlopen(r, timeout=timeout)

                encoding = http_response.headers.get("Content-Encoding")
                if encoding and "gzip" in encoding:
                    data = gzip.decompress(http_response.read()).decode(
                        encoding="utf-8", errors="ignore"
                    )
                elif encoding is None:
                    data = http_response.read().decode(encoding="utf-8", errors="ignore")
                else:
                    raise NotImplementedError

                result["body"] = data

            except error.HTTPError as e:
                set_error(self.error_types['connection_error'], website.url)
                return
            except (error.URLError,
                    InvalidURL,
            ) as e:
                set_error(self.error_types['url_error'], website.url)
                return
            except (SocketTimeoutError,
                    TimeoutError,
            ) as e:
                set_error(self.error_types['timeout_error'], website.url)
                return

            return result
        else:
            set_error(self.error_types['path_error'], website.url)

    def parse_page(self, page):
        url_hash = hashlib.sha256(page["url"].encode('utf-8')).hexdigest()

        soup = BeautifulSoup(page["body"], "html.parser")

        file_location = self.data_folder + page["category"] + "/" + url_hash + ".txt"
        try:
            title_text = soup.title.string
        except:
            set_error(self.error_types['title_error'], page["url"])
        
        body_text = str(extract_plain_text(str(soup.body), main_content=True, preserve_formatting=False))
        body_text = re.sub("[^\x00-\x7F]+", "", body_text)
        if (not body_text.isspace()) and body_text != "" and body_text != "None":
            if (not title_text.isspace()):

                page["body"] = body_text
                page["title"] = title_text

                with open(file_location, 'w') as json_file:
                    json.dump(page, json_file, indent=4, sort_keys=True)
            else:
                set_error(self.error_types['title_error'], page["url"])
        else:
            set_error(self.error_types['body_error'], page["url"])

    def handle_future(self, result):
        if result.result():
            page = result.result()
            self.parse_page(page)

    def run(self, website_list):
        for _, target_website in website_list.iterrows():
            try:
                job = self.pool.submit(
                    self.load_url, target_website, timeout=2
                )
                job.add_done_callback(self.handle_future)
            except Empty:
                return
            except Exception as e:
                print(e)
        
def websites_processed(df, data_path):
    try:
        list_of_files = glob.glob(data_path + "/" + sys.argv[1] + "/*")
        latest_file = max(list_of_files, key=os.path.getctime)

        with open(latest_file) as f:
            data = json.load(f)

        index_file = df.loc[df['url'] == data['url']]
    
        return index_file.index[0] + 1
    except:
        return 0

def set_error(error_type, url):
    field_names = ['Error', 'Url']
    item = {'Error': error_type, 'Url': url}

    with open(str(sys.argv[2]) + "/error_index_" + str(sys.argv[1]) + ".csv", 'a') as error_file:
        dict_object = csv.DictWriter(error_file, fieldnames=field_names)
        dict_object.writerow(item)

def main():
    data_path = sys.argv[1]
    df = pd.read_csv(data_path + "data_dmoz.csv", index_col="Unnamed: 0")
    df = df.loc[df['category'] == str(sys.argv[2])]
    website_list = df.loc[websites_processed(df, data_path):]

    get = GetWebsite(target_folder=data_path)
    get.run(website_list)

if __name__ == "__main__":
    main()
