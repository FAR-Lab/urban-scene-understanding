import detect_shadow_deweighting as dsd
import subprocess
import fire
import shlex
import re
import os
import parsing as p
import extract_clips as ec
import logging

logging.basicConfig(level=logging.INFO)

def intersection_detection(source): 
    PROJECT_PATH = "/share/ju/xcpedestrian/urban-scene-understanding"
    YOLOV7_WEIGHTS = os.path.join(PROJECT_PATH, "yolov7/pretrained/yolov7x.pt")
    INTERSECTION_COCO_CLASSES = [9, 11]
    TOP_LEVEL_DIR = os.path.join("intersection_detection")
    CDNET_WEIGHTS = os.path.join(PROJECT_PATH, "cdnet.pt")

    stripped_source_path = re.sub(r'^[/\\.]+', '', source)
    logging.info(f'Processing source: {stripped_source_path}')

    check_label_output_dir(stripped_source_path, f"{PROJECT_PATH}/scripts", TOP_LEVEL_DIR, "_coco", "_cdnet")

    traffic_light_and_shadow_deweighting_cmd = f'python detect_shadow_deweighting.py --source {source} --weights {YOLOV7_WEIGHTS} --classes {" ".join(map(str, INTERSECTION_COCO_CLASSES))} --save-conf --save-txt --project {TOP_LEVEL_DIR} --name {os.path.splitext(os.path.basename(source))[0]}_coco'
    logging.info('Running traffic light and shadow deweighting')
    subprocess.run(shlex.split(traffic_light_and_shadow_deweighting_cmd))

    intersection_detection_cmd = f'python {PROJECT_PATH}/yolov7/detect.py --source {source} --weights {CDNET_WEIGHTS} --save-txt --save-conf --project {TOP_LEVEL_DIR} --name {os.path.splitext(os.path.basename(source))[0]}_cdnet'
    logging.info('Running intersection detection')
    subprocess.run(shlex.split(intersection_detection_cmd))

    logging.info('Parsing periods')
    periods = p.parse(stripped_source_path)
    logging.info(f'Parsed periods: {periods}')

    logging.info('Extracting clips')
    ec.extract_clips(f"{PROJECT_PATH}/{stripped_source_path}", periods, f"{PROJECT_PATH}/intersection_clips/{os.path.splitext(os.path.basename(source))[0]}")

def check_label_output_dir(source, proj_path, top_level_dir, coco_suffix, cdnet_suffix): 
    coco = os.path.exists(os.path.join(proj_path, top_level_dir, f"{os.path.splitext(os.path.basename(source))[0]}{coco_suffix}"))
    if coco: 
        logging.info(f"COCO label output directory for {source} already exists")

    cdnet = os.path.exists(os.path.join(proj_path, top_level_dir, f"{os.path.splitext(os.path.basename(source))[0]}{cdnet_suffix}"))
    if cdnet: 
        logging.info(f"CDNET label output directory for {source} already exists")
    
    if coco or cdnet: 
        erase = input("Erase existing label output directories? (y/n): ")
        if erase.lower() == "y":
            if coco: 
                os.system(f"rm -rf {os.path.join(proj_path, top_level_dir, f'{os.path.splitext(os.path.basename(source))[0]}{coco_suffix}')}")
                logging.info(f"Erased {os.path.join(proj_path, top_level_dir, f'{os.path.splitext(os.path.basename(source))[0]}{coco_suffix}')}")
            if cdnet: 
                os.system(f"rm -rf {os.path.join(proj_path, top_level_dir, f'{os.path.splitext(os.path.basename(source))[0]}{cdnet_suffix}')}")
                logging.info(f"Erased {os.path.join(proj_path, top_level_dir, f'{os.path.splitext(os.path.basename(source))[0]}{cdnet_suffix}')}")


def main():
    fire.Fire(intersection_detection)

if __name__ == '__main__': 
    main()
