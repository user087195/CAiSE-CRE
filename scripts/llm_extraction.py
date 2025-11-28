import os
import json
import argparse
from vllm import LLM, SamplingParams
from xml2text import get_text_and_tables
import xml.etree.ElementTree as ET


BASE_DIR = "/home/USER/caise26/"


with open('../prompts/extraction_prompt.txt') as infile:

	extract_text = infile.read()


parser = argparse.ArgumentParser(description="Set output directory and visible devices.")
parser.add_argument("--outdir", type=str, help="Output path")
parser.add_argument("--devices", type=str, help="CUDA visible devices: 0,1,2,3")
parser.add_argument("--mode", type=str, default='val', help="Validation or test data: val or test")
#parser.add_argument("--ocr_engine", type=str, default='grobid', help="OCR engine: grobid, nougat, cermine")

args = parser.parse_args()

n_devices = len(args.devices.split(','))
os.environ["CUDA_VISIBLE_DEVICES"] = args.devices


OUTDIR = args.outdir
if not os.path.exists(OUTDIR):
    os.makedirs(OUTDIR)

EXISTING = [f for f in os.listdir(OUTDIR)]

sampling_params = SamplingParams(temperature=0.1, 
                                 max_tokens=4096)
llm = LLM(model="deepseek-ai/DeepSeek-R1-Distill-Llama-70B", 
          tensor_parallel_size=n_devices)

suffixes = {'grobid': '.grobid.tei.xml', 'grobid-alt': '.grobid.tei.xml', 'nougat': '.mmd', 'cermine': '.cermxml', 'pymupdf': '.pdf.txt', 'pymupdf4llm': '.md'}

for t in [0.1]:#, None]:
    if t in [0.1, 0.5]:
        sampling_params = SamplingParams(temperature=t, max_tokens=5120)
    else:
        sampling_params = SamplingParams(max_tokens=5120)
        
    for ocr_engine in ['nougat', 'cermine', 'pymupdf', 'pymupdf4llm']:

        if ocr_engine == 'grobid-alt':
            PATH2DIR = f"{BASE_DIR}/data/{args.mode}/grobid_out_pubmed"
        else:
            PATH2DIR = f"{BASE_DIR}/data/{args.mode}/{ocr_engine}_out"
        file_suff = suffixes[ocr_engine]

        xmls = [fname for fname in os.listdir(PATH2DIR) if fname.endswith(file_suff)]

        if ocr_engine == 'grobid':
            texts  = [get_text_and_tables(os.path.join(f'{BASE_DIR}/data/{args.mode}/grobid_out_pubmed', fname))[0] for fname in xmls]
            tables = [get_text_and_tables(os.path.join(f'{BASE_DIR}/data/{args.mode}/grobid_out_pubmed', fname))[1] for fname in xmls]
            contexts = ['\n'.join(text) + '\n' + '\n'.join(table) for text, table in zip(texts, tables)]
        elif ocr_engine == 'grobid-alt':
            trees = [ET.parse(os.path.join(f'{BASE_DIR}/data/{args.mode}/grobid_out_pubmed', fname)) for fname in xmls]
            xml_strs = [ET.tostring(tree.getroot(), encoding='utf-8', method='text') for tree in trees]
            contexts = [xml_str.decode('utf-8') for xml_str in xml_strs]
        elif ocr_engine == 'nougat':
            contexts = []
            for fname in xmls:
                with open(os.path.join(f'{BASE_DIR}/data/{args.mode}/nougat_out', fname), 'r') as f:
                    contexts.append(f.read())
        elif ocr_engine == 'cermine':
            trees = [ET.parse(os.path.join(f'{BASE_DIR}/data/{args.mode}/cermine_out', fname)) for fname in xmls]
            xml_strs = [ET.tostring(tree.getroot(), encoding='utf-8', method='text') for tree in trees]
            contexts = [xml_str.decode('utf-8') for xml_str in xml_strs]
        elif ocr_engine == 'pymupdf':
            contexts = []
            for fname in xmls:
                with open(os.path.join(f'{BASE_DIR}/data/{args.mode}/pymupdf_out', fname), 'r') as f:
                    contexts.append(f.read())
        elif ocr_engine == 'pymupdf4llm':
            contexts = []
            for fname in xmls:
                with open(os.path.join(f'{BASE_DIR}/data/{args.mode}/pymupdf4llm_out', fname), 'r') as f:
                    contexts.append(f.read())

        extraction_prompts = [extract_text + f"\n\n```{context}```" for context in contexts]

        for i in range(0, 5):

            try:
                print(f"Running extraction for {ocr_engine} with temperature {t} and run {i+1}, {len(extraction_prompts)} prompts in total.")
                outputs = llm.generate(extraction_prompts, sampling_params)

                for fname, output in zip(xmls, outputs):
                    with open(os.path.join(OUTDIR, fname.replace(file_suff, f'_preds_{ocr_engine}_temp={t}_run{i+1}.txt')), 'w') as out1:
                        out1.write(output.outputs[0].text)
                
            except:
                print("Failed extracting relations")

