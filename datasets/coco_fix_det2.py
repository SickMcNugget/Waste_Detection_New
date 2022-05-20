import json
import os
import argparse

parser = argparse.ArgumentParser(description="Correct coco json files in a directory")
parser.add_argument('path', type=str, help='The path to correct')

args = parser.parse_args()

if not os.path.exists(args.path):
    print(f"Error: {args.path} is not a valid path")
else:
    data_dirs = {}
    data_dirs['train'] = os.path.join(args.path, "train/")
    data_dirs['valid'] = os.path.join(args.path, "valid/")
    data_dirs['test'] = os.path.join(args.path, "test/")

    for k in data_dirs:
        if os.path.exists(data_dirs[k]):
            print(f"Correcting annotations from the {k} directory...")
            ann_file = f"{data_dirs[k]}_annotations.coco.json"

            with open(ann_file, 'r') as json_file:
                json_data = json.load(json_file)

            category_name = json_data['categories'][0]['name']
            json_data['categories'] = json_data['categories'][1:]
            
            for category in json_data['categories']:
                if category['supercategory'] == category_name:
                    category['supercategory'] = 'none'

            print(f"Removed all references to category {category_name} from {k} directory\n")

            with open(ann_file, "w") as json_file:
                json.dump(json_data, json_file, indent=4)
