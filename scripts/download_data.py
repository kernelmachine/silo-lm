import os
import json
import time
import re
import numpy as np
import csv
from tqdm import tqdm
from functools import partial
import argparse
import subprocess
from datasets import load_dataset

from multiprocessing import Pool

DATA_DIR = "data"

def main(args):
    if not os.path.isdir(DATA_DIR):
        os.mkdir(DATA_DIR)

    if args.subset=="the-pile":
        for split in args.split.split(","):
            assert split in ["train", "val", "test"], split
            download_the_pile(split)
    elif args.subset=="cc-news":
        download_cc_news()
    elif args.subset=="MIMIC_III":
        download_mimic()
    elif args.subset=="amazon":
        download_amazon()
    else:
        raise NotImplementedError()

def download_the_pile(split):

    if not os.path.isdir(os.path.join(DATA_DIR, "the-pile")):
        os.mkdir(os.path.join(DATA_DIR, "the-pile"))

    if split in ["val", "test"]:
        command = "wget -O data/the-pile/{}.jsonl.zst https://the-eye.eu/public/AI/pile/{}.jsonl.zst".format(split, split)
        ret_code = subprocess.run([command],
                                shell=True,
                                #stdout=subprocess.DEVNULL,
                                # #stderr=subprocess.STDOUT
                                )
        if ret_code.returncode != 0:
            print("Failed to download the data")
            exit()
        else:
            print("Successfully downloaded the data")

    elif split=="train":
        subset_to_id = {
            "Enron_Emails": "1zxWJxqseOJ7FB6KtVhyA8NJhwpD3JQiw",
            "Books3": "1NuqcsRXUQcupsafWzEe6o7_MjR4tesIA",
            "Github": "1v3BFvtniOjN1BCOWjpKWjkbOGv5jwjQ2",
            "NIH_ExPorter": "1pqXJRdvxDPbJXaJBvOs1kNSSb5qXuOdD",
            "Wikipedia_\(en\)": "1WRyuCaXkYDV8JlJ2JjWQLcIok7WY1IKA"
        }

        for subset, _id in subset_to_id.items():
            target_file = os.path.join(DATA_DIR, "the-pile", "train-gz", f"{subset}.tar.gz")
            download_gdrive_file(_id, target_file)
            command = "tar -vxzf %s -C %s; rm %s" % (target_file, os.path.join(DATA_DIR, "the-pile", "train-gz"), target_file)
            ret_code = subprocess.run([command], shell=True)
            if ret_code.returncode != 0:
                print(f"Failed to download the {subset} data")
                exit()
            else:
                print(f"Successfully downloaded the {subset} data")

    else:
        raise NotImplementedError()

def download_amazon():
    try:
        orig_data_dir = os.path.join(DATA_DIR, "raw_amazon")
        data_dir = os.path.join(DATA_DIR, "amazon")

        if not os.path.exists(orig_data_dir):
            os.mkdir(orig_data_dir)
        if not os.path.exists(data_dir):
            os.mkdir(data_dir)

        # 46 configs
        configs = ['Wireless_v1_00', 'Watches_v1_00', 'Video_Games_v1_00', 'Video_DVD_v1_00', 'Video_v1_00', 'Toys_v1_00', 'Tools_v1_00', 'Sports_v1_00', 'Software_v1_00', 'Shoes_v1_00', 'Pet_Products_v1_00', 'Personal_Care_Appliances_v1_00', 'PC_v1_00', 'Outdoors_v1_00', 'Office_Products_v1_00', 'Musical_Instruments_v1_00', 'Music_v1_00', 'Mobile_Electronics_v1_00', 'Mobile_Apps_v1_00', 'Major_Appliances_v1_00', 'Luggage_v1_00', 'Lawn_and_Garden_v1_00', 'Kitchen_v1_00', 'Jewelry_v1_00', 'Home_Improvement_v1_00', 'Home_Entertainment_v1_00', 'Home_v1_00', 'Health_Personal_Care_v1_00', 'Grocery_v1_00', 'Gift_Card_v1_00', 'Furniture_v1_00', 'Electronics_v1_00', 'Digital_Video_Games_v1_00', 'Digital_Video_Download_v1_00', 'Digital_Software_v1_00', 'Digital_Music_Purchase_v1_00', 'Digital_Ebook_Purchase_v1_00', 'Camera_v1_00', 'Books_v1_00', 'Beauty_v1_00', 'Baby_v1_00', 'Automotive_v1_00', 'Apparel_v1_00', 'Digital_Ebook_Purchase_v1_01', 'Books_v1_01', 'Books_v1_02']

        cnt = 0
        n_tokens_tot = 0
        start_time = time.time()

        with Pool() as pool:
            for n in pool.imap_unordered(partial(process_data, data_dir=orig_data_dir), configs):
                cnt += 1
                n_tokens_tot += n
                print ("Finish %d/%d (%dM tokens, %dmin)" % (
                    cnt, len(configs), n_tokens_tot/1000000, (time.time()-start_time)/60
                ))

        train_path = os.path.join(data_dir, "train.jsonl")
        val_path = os.path.join(data_dir, "val.jsonl")
        test_path = os.path.join(data_dir, "test.jsonl")

        if os.path.exists(train_path):
            os.remove(train_path)
        if os.path.exists(val_path):
            os.remove(val_path)
        if os.path.exists(test_path):
            os.remove(test_path)

        for fn in tqdm(sorted(os.listdir(orig_data_dir))):
            lines = []

            # cut at 1M lines
            with open(os.path.join(orig_data_dir, fn)) as f:
                for line in f:
                    lines.append(line)
                    if len(lines)==500000:
                        break

            # 1% as val, 1% as test, 98% as train
            val_size = len(lines) // 100
            with open(val_path, "a+") as f:
                for line in lines[:val_size]:
                    f.write(line)
            with open(test_path, "a+") as f:
                for line in lines[val_size:2*val_size]:
                    f.write(line)
            with open(train_path, "a+") as f:
                for line in lines[2*val_size:]:
                    f.write(line)
    except Exception:
        """For some reason, amazon corpus in Huggingface does not work anymore, so added a way to
        download from Google Drive"""

        target_file = os.path.join(DATA_DIR, "amazon", "amazon.tar.gz")
        download_gdrive_file("1tQCPfKGLV9WRjA_9y7YO3LCJjaFYlx04", target_file)
        assert os.path.exists(target_file)
        command = "tar -vxzf %s -C %s; rm %s" % (target_file, os.path.join(DATA_DIR, "amazon"), target_file)
        ret_code = subprocess.run([command], shell=True)
        if ret_code.returncode != 0:
            print("Failed to download the data")
            exit()
        else:
            print("Successfully downloaded the data")


def process_data(config, data_dir):
    data = load_dataset("amazon_us_reviews", config, streaming=True, split="train")

    n_tokens = 0
    with open(os.path.join(data_dir, "{}.jsonl".format(config)), "w") as f:
        for dp in data:
            text = dp["review_body"].replace("<br />", "\n")
            n_tokens += len(text.split())
            f.write(json.dumps({"text": text})+"\n")
    return n_tokens

def download_mimic():
    in_file = os.path.join(DATA_DIR, "NOTEEVENTS.csv")
    assert os.path.join(in_file), in_file

    col = None
    n_lines = 0
    n_tokens = 0

    def replace_newlines(text):
        pattern = re.compile(r"(?<!\n)\n(?!\n)")
        replaced_text = re.sub(pattern, " ", text)
        return replaced_text

    # about 2M lines (500M tokens)
    texts = []

    with open(in_file, "r") as f:
        for row in csv.reader(f):
            if col is None:
                col = row
            else:
                dp = {k:v for k, v in zip(col, row)}
                texts.append(replace_newlines(dp["TEXT"]))

    print ("# lines = %d, # tokens = %d" % (
        len(texts), np.sum([len(text.split()) for text in texts])
    ))

    val_size = len(texts) // 100

    if not os.path.isdir(os.path.join(DATA_DIR, "MIMIC_III")):
        os.mkdir(os.path.join(DATA_DIR, "MIMIC_III"))

    with open(os.path.join(DATA_DIR, "MIMIC_III", "val.jsonl"), "w") as f:
        for text in texts[:val_size]:
            f.write(json.dumps({"text": text})+"\n")
    with open(os.path.join(DATA_DIR, "MIMIC_III", "test.jsonl"), "w") as f:
        for text in texts[val_size:2*val_size]:
            f.write(json.dumps({"text": text})+"\n")
    with open(os.path.join(DATA_DIR, "MIMIC_III", "train.jsonl"), "w") as f:
        for text in texts[2*val_size:]:
            f.write(json.dumps({"text": text})+"\n")

def download_cc_news():
    cc_news = load_dataset('cc_news', split="train")

    paragraphs = []
    for dp in cc_news:
        paragraphs.append(dp["text"])

    n_test_lines = 2000
    np.random.seed(2023)
    indices = np.random.permutation(range(len(paragraphs)))
    valid_indices = set(indices[:n_test_lines])
    test_indices = set(indices[n_test_lines:2*n_test_lines])

    output_dir = os.path.join(DATA_DIR, "cc-news")
    if not os.path.exists(output_dir):
        os.mkdir(output_dir)
    with open(os.path.join(output_dir, "train.jsonl"), "w") as f_train:
        with open(os.path.join(output_dir, "val.jsonl"), "w") as f_val:
            with open(os.path.join(output_dir, "test.jsonl"), "w") as f_test:
                for i, text in enumerate(paragraphs):
                    if i in valid_indices:
                        f_val.write(json.dumps({"text": text.strip()})+"\n")
                    elif i in test_indices:
                        f_test.write(json.dumps({"text": text.strip()})+"\n")
                    else:
                        f_train.write(json.dumps({"text": text.strip()})+"\n")


def download_gdrive_file(_id, dest):
    if os.path.exists(dest):
        print ("[Already exists] Skipping", dest)
        print ("If you want to download the file in another location, please specify a different path")
        return

    if "/" in dest:
        dest_dir = "/".join(dest.split("/")[:-1])
        if not os.path.isdir(dest_dir):
            os.makedirs(dest_dir)
    else:
        dest_dir = "."

    if _id.startswith("https://"):
        command = """wget -O %s %s""" % (dest, _id)
    else:
        command = """wget --load-cookies /tmp/cookies.txt "https://docs.google.com/uc?export=download&confirm=$(wget --quiet --save-cookies /tmp/cookies.txt --keep-session-cookies --no-check-certificate 'https://docs.google.com/uc?export=download&id=%s' -O- | sed -rn 's/.*confirm=([0-9A-Za-z_]+).*/\\1\\n/p')&id=%s" -O %s && rm -rf /tmp/cookies.txt""" % (_id, _id, dest)

    ret_code = subprocess.run([command], shell=True)
    if ret_code.returncode != 0:
        print("Download {} ... [Failed]".format(dest))
    else:
        print("Download {} ... [Success]".format(dest))

    if dest.endswith(".zip"):
        command = """unzip %s -d %s && rm %s""" % (dest, dest_dir, dest)

        ret_code = subprocess.run([command], shell=True)
        if ret_code.returncode != 0:
            print("Unzip {} ... [Failed]".format(dest))
        else:
            print("Unzip {} ... [Success]".format(dest))


if __name__=='__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--subset",
                        type=str,
                        default=None,
                        choices=["the-pile", "cc-news", "MIMIC_III", "amazon"])
    parser.add_argument("--split",
                        type=str,
                        default="train,val,test")


    args = parser.parse_args()
    print (args)
    main(args)
