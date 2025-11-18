# SPDX-FileCopyrightText: Copyright (c) 2024 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import requests
import os
import re
from concurrent.futures import ThreadPoolExecutor, as_completed
import pandas as pd
import argparse

# Function to create a valid filename
def create_valid_filename(s):
    # Remove invalid characters and replace spaces with underscores
    s = re.sub(r'[^\w\-_\. ]', '', s)
    return s.replace(' ', '_')


def csv_to_txt(csv_file="./data/gear-store.csv"):
    df = pd.read_csv(csv_file)
    os.makedirs('./data/product', exist_ok=True)
    # Iterate through each row in the DataFrame
    for index, row in df.iterrows():
        # Create filename using name, category, and subcategory
        filename = f"{create_valid_filename(row['name'])}_{create_valid_filename(row['category'])}_{create_valid_filename(row['subcategory'])}.txt"

        print(f"Creating file {filename}, current index {index}")
        # Full path for the file
        filepath = os.path.join('./data/product', filename)

        # Create the content for the file
        content = f"Name: {row['name']}\n"
        content += f"Category: {row['category']}\n"
        content += f"Subcategory: {row['subcategory']}\n"
        content += f"Price: ${row['price']}\n"
        content += f"Description: {row['description']}\n"

        # Write the content to the file
        with open(filepath, 'w', encoding='utf-8') as file:
            file.write(content)

    print(f"Created {len(df)} files in ./data/product")

def get_health(url: str):
    health_url = f'{url}/health'
    headers = {
        'accept': 'application/json'
    }
    response = requests.get(health_url, headers=headers)
    return response.status_code

def ingest_manuals( url: str, directory_path='./data/manuals_pdf'):
    document_url = f'{url}/documents'
    for filename in os.listdir(directory_path):
        # Check if the file is a PDF
        if filename.endswith('.pdf'):
            file_path = os.path.join(directory_path, filename)

            # Open the file in binary mode and send it in a POST request
            with open(file_path, 'rb') as file:
                files = {'file': file}
                response = requests.post(document_url, files=files)

            # Print the response from the server
            print(f'Uploaded {filename}: {response.status_code}')

def ingest_faqs(url: str, filename = "./data/FAQ.pdf"):
    document_url = f'{url}/documents'
    with open(filename, 'rb') as file:
        files = {'file': file}
        try:
            response = requests.post(document_url, files=files)
            print(f'Uploaded {filename}: {response.status_code}')
            return response.status_code == 200
        except requests.exceptions.RequestException as e:
            print(f"Request failed for {filename}: {e}")
            return False


#Skipping get the list of documents


def ingest_csvs(url: str, directory_path='./data/product', max_workers=5):
    filepaths = [
        os.path.join(directory_path, filename)
        for filename in os.listdir(directory_path) if filename.endswith(".txt")
    ]
    successfully_ingested = []
    failed_ingestion = []
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        future_to_file = {
            executor.submit(ingest_faqs, url, filepath): filepath
            for filepath in filepaths
        }

        for future in as_completed(future_to_file):
            filepath = future_to_file[future]
            try:
                if future.result():
                    print(
                        f"Successfully Ingested {os.path.basename(filepath)}")
                    successfully_ingested.append(filepath)
                else:
                    print(f"Failed to Ingest {os.path.basename(filepath)}")
                    failed_ingestion.append(filepath)
            except Exception as e:
                print(
                    f"Exception occurred while ingesting {os.path.basename(filepath)}: {e}"
                )
                failed_ingestion.append(filepath)

    print(f"Total files successfully ingested: {len(successfully_ingested)}")
    print(f"Total files failed ingestion: {len(failed_ingestion)}")

#Skipping Document delete

if __name__ == "__main__":
    # Set up the argument parser
    parser = argparse.ArgumentParser(description='Database connection parameters.')
    parser.add_argument('--host', type=str, default='localhost', help='Database host')
    parser.add_argument('--port', type=str, default='8086', help='Database port')
    args = parser.parse_args()

    url =  f'http://{args.host}:{args.port}'

    health_code = get_health(url)

    print(health_code)

    ingest_manuals(url=url)
    ingest_faqs(url=url)
    csv_to_txt()
    ingest_csvs(url=url)
