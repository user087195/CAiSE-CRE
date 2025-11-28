import re
import copy
import os, json
from collections import defaultdict
from wordfreq import zipf_frequency
import xml.etree.ElementTree as ET
from abbreviations import schwartz_hearst



def get_paper_abbreviations(fname, mode, engine, file_suff):

    if not os.path.exists(f'../code/ecis25_data/{mode}/{engine}_out/{fname}{file_suff}'):
        return {}
    else:
        abbreviations = {}
        if engine == 'grobid':
            tree = ET.parse(f'../code/ecis25_data/{mode}/{engine}_out/{fname}{file_suff}')
            xml_str = ET.tostring(tree.getroot(),
                                  encoding='utf-8',
                                  method='text')
            xml_str = xml_str.decode('utf-8')
            texts = xml_str.split('. ')

            for t in texts:

                pairs = schwartz_hearst.extract_abbreviation_definition_pairs(doc_text=t,
                                                                              most_common_definition=True)

                for k_pairs, v_pairs in pairs.items():

                    if not k_pairs in abbreviations.keys():
                        abbreviations[k_pairs] = v_pairs

        return abbreviations



def get_paper_figure_mapping(pred_dir, suffix):

    paper_figures = {}

    for fname in os.listdir(pred_dir):
        full_fname = fname
        fname = fname.replace(suffix, '')
        fname = '_'.join(fname.split('_')[:-3])
        if not fname in paper_figures.keys():
            paper_figures[fname] = [full_fname]
        else:
            paper_figures[fname].append(full_fname)

    return paper_figures



def is_number_regex(s):
    number_regex = re.compile(r'^[-+]?\d*\.?\d+$')
    return bool(number_regex.match(s))


def _normalize_name(name):

    # Check it is not a string continue
    if type(name) == float or name is None:
        return None

    # Long strings could be caused by errors in the OCR or merged boxes
    if len(name.split()) > 15:
        return None

    if all([x in '-.,01234567890 ' for x in name]):
        return None

    # A variable name cannot be purely numeric
    if is_number_regex(name):
        return None

    # Preserve clean variable codes
    if re.fullmatch(r"[A-Za-z0-9]+", name):
        return name

    # Replace underscores by whitespaces
    name = name.replace('_', ' ')

    # Normalize sub- and superscripts (₀ to ₉ → 0 to 9)
    subscript_map = str.maketrans("₀₁₂₃₄₅₆₇₈₉", "0123456789")
    superscript_map = str.maketrans("⁰¹²³⁴⁵⁶⁷⁸⁹", "0123456789")

    name = name.translate({**subscript_map, **superscript_map})

    # Insert a space between any letter and parantheses (A–Z, case-insensitive)
    name = re.sub(r"([A-Za-z])\(", r"\1 (", name)

    return name


def clean_variable_name(output_data):
    for i, rel in enumerate(output_data):
        new_cause  = _normalize_name(rel['cause'])
        new_effect = _normalize_name(rel['effect'])
        output_data[i]['cause'] = new_cause
        output_data[i]['effect'] = new_effect
    return output_data



def to_zero_point_x(x):
    # Handles cases where path coefficients miss the leading 0.
    if x >= 10:
        return float('0.' + str(int(x)))
    elif x <= -10:
        return float(str(int(x)).replace('-', '-0.'))
    else:
        return x



def get_abbreviation_map_vars(var_preds):

    abbreviations = {}

    for t in var_preds:
        pairs = schwartz_hearst.extract_abbreviation_definition_pairs(doc_text=t,
                                                                      most_common_definition=True)
        for k, v in pairs.items():
            if not k in abbreviations.keys():
                abbreviations[k] = v

    return abbreviations




def clean_relations_remove_indicators(relations, only_hyp=False):
    # regex for indicator tokens such as: A1, BI4, PEOU3, ABC12
    indicator_pattern = re.compile(r"\b[A-Za-z]{1,5}\d{1,3}\b")

    cleaned = []

    for rel in relations:
        cause  = rel.get("cause", "")
        effect = rel.get("effect", "")
        beta   = rel.get("beta", "")
        hyp    = rel.get("hypothesis_id", "")

        if not cause or not effect or cause == effect:
            continue

        # if cause OR effect contains an indicator token → skip
        if indicator_pattern.search(cause) or indicator_pattern.search(effect):
            continue

        if len(cause) < 4 or len(effect) < 4:
            continue

        if (not beta is None) and (type(beta) != float) and (not is_number_regex(str(beta))):
            continue

        if only_hyp:
            if beta is None and hyp is None:
                continue

        cleaned.append(rel)

    return cleaned



def apply_mapping_vars(output_data):

    mapping_dict = get_abbreviation_map_vars(set([x['cause'] for x in output_data] + [x['effect'] for x in output_data]))

    for k, v in mapping_dict.items():

        for idx, rel in enumerate(output_data):
            if f'({k.lower()})' in rel['cause'].lower():
                output_data[idx]['cause'] = output_data[idx]['cause'].replace(f'({k})', '').strip()

            if f'({k.lower()})' in rel['effect'].lower():
                output_data[idx]['effect'] = output_data[idx]['effect'].replace(f'({k})', '').strip()

            if 'moderator' in rel.keys() and rel['moderator']:
                if f'({k.lower()})' in rel['moderator'].lower():
                    output_data[idx]['moderator'] = output_data[idx]['moderator'].replace(f'({k})', '').strip()

    return output_data



def apply_abbreviation_mapping(output_data, abbreviations):
    for i, rel in enumerate(output_data):
        if rel['cause'].lower() in abbreviations:
            output_data[i]['cause'] = abbreviations[rel['cause'].lower()]
        if rel['effect'].lower() in abbreviations:
            output_data[i]['effect'] = abbreviations[rel['effect'].lower()]
        if rel['moderator'] and rel['moderator'].lower() in abbreviations:
            output_data[i]['moderator'] = abbreviations[rel['moderator']]
    return output_data


def _parentheses(s):

    groups = re.findall(r"\([^()]*\)", s)
    to_remove = []

    for g in groups:
        inner = g[1:-1]

        # --- Condition 1: inner is only uppercase letters ---
        if inner and re.fullmatch(r"[A-Z]+", inner):
            to_remove.append(g)
            continue

        # --- Condition 2: contains KEY = number anywhere ---
        if re.search(r"\S+\s*=\s*\d+(?:\.\d+)?", inner):
            to_remove.append(g)
            continue

        # --- Condition 3: more than 50% digits or . * - % ---
        special_chars = set("0123456789.*-%")
        if inner:
            count = sum(ch in special_chars for ch in inner)
            if count / len(inner) > 0.5:
                to_remove.append(g)
                continue

    # Remove all identified groups from the original string
    for g in to_remove:
        s = s.replace(g, "")

    return s


def clean_parantheses(output_data):
    for i, rel in enumerate(output_data):
        new_cause = _parentheses(rel['cause'])
        new_effect = _parentheses(rel['effect'])
        output_data[i]['cause'] = new_cause
        output_data[i]['effect'] = new_effect
    return output_data



def _assignments(text):
    """
    Removes: <last-token-before-equals> = <number>%?
    Keeps everything before that last token.
    """
    # Number pattern (signed int/float/scientific + optional %)
    num = r'[+-]?(?:\d*\.\d+|\d+\.?|\.\d+)(?:[eE][+-]?\d+)?%?'

    # Match:
    #   last non-space token before '='   →   \S+
    #   optional whitespace               →   \s*
    #   '='                               →   =
    #   number pattern                    →   num
    pattern = rf'\S+\s*=\s*{num}'

    # Perform replacement
    out = re.sub(pattern, '', text)

    # Normalize spacing but preserve trailing spaces before removal point
    return " ".join(out.split()) if out.strip() else ''


def remove_assignments(output_data):
    for i, rel in enumerate(output_data):
        new_cause = _assignments(rel['cause'])
        new_effect = _assignments(rel['effect'])
        output_data[i]['cause'] = new_cause
        output_data[i]['effect'] = new_effect
    return output_data



def remove_indicator(output_data):
    for i, rel in enumerate(output_data):
        if rel['cause'] and rel['cause'].strip()[-1].isdigit() and len(rel['cause'].split()) == 1:
            output_data[i]['cause'] = None
        if rel['effect'] and rel['effect'].strip()[-1].isdigit() and len(rel['effect'].split()) == 1:
            output_data[i]['effect'] = None
    return output_data




def _edge_floats_or_percent(s):
    # float with optional %: 12, 12.34, .5, 5., with optional sign, and optional %
    float_pct = r"[+-]?(?:\d+(?:\.\d*)?|\.\d+)%?"
    return re.sub(rf"^{float_pct}|{float_pct}$", "", s).strip()


def remove_edge_floats_or_percent(output_data):
    for i, rel in enumerate(output_data):
        new_cause = _edge_floats_or_percent(rel['cause'])
        new_effect = _edge_floats_or_percent(rel['effect'])
        output_data[i]['cause'] = new_cause
        output_data[i]['effect'] = new_effect
    return output_data



def standardize_relations(data):
    output_data = []
    for idx, x in enumerate(data):

        if x['label_text']:
            res = keyword_search(x['label_text'])
        else:
            res = keyword_search('')

        var_name1 = x['source_node_text']
        var_name2 = x['target_node_text']

        if not var_name1 or not var_name2:
            continue

        tmp_rel = {'cause': var_name1,
                   'effect': var_name2,
                   'moderator': None,
                   'beta': None,
                   'p': None,
                   'hypothesis_id': None
                   }


        # Add extracted beta value
        if len(res['betas']) > 0:
            tmp_rel['beta'] = float(res['betas'][0][-1])
        elif len(res['numbers']) > 0:
            num = res['numbers'][0]
            if '.' in num:
                tmp_rel['beta'] = float(num)
            else:
                tmp_rel['beta'] = to_zero_point_x(float(num))

        # Add extracted hypothesis id
        if len(res['hypotheses']) > 0:
            tmp_rel['hypothesis_id'] = res['hypotheses'][0]

        # Add extracted p-value
        if len(res['p_values']) > 0:
            if len(res['p_values'][0]) == 2:
                tmp_rel['p'] = res['p_values'][0][-1]
            elif len(res['p_values'][0]) == 3:
                tmp_rel['p'] = res['p_values'][0][1] + res['p_values'][0][2]
        elif len(res['stars']) > 0:
            tmp_rel['p'] = res['stars'][0]


        output_data.append(tmp_rel)

    return output_data




def run_cleaning(path2preds,
                 file_suffix,
                 path2save,
                 file_suffix_out,
                 mode='val',
                 standardize=True):

    for fname in sorted(os.listdir(path2preds)):

        if not fname.endswith(file_suffix):
            continue

        print(fname)

        paper_name = '_'.join(fname.replace(file_suffix, "").split('_')[:-3])
        abbreviations = get_paper_abbreviations(paper_name,
                                                mode,
                                                'grobid',
                                                '.grobid.tei.xml')

        print(abbreviations)
        with open(os.path.join(path2preds, fname)) as infile:
            data = json.load(infile)

        if standardize:
            output_data = standardize_relations(data)
        else:
            output_data = data

        cur_vars = set([x['cause'] for x in output_data] + [x['effect'] for x in output_data])
        print(cur_vars)


        output_data = clean_variable_name(output_data)

        output_data = apply_abbreviation_mapping(output_data, abbreviations)

        output_data = apply_mapping_vars(output_data)

        output_data = clean_parantheses(output_data)

        output_data = remove_assignments(output_data)

        output_data = remove_edge_floats_or_percent(output_data)

        output_data = clean_relations_remove_indicators(output_data, only_hyp=False)

        cur_vars = set([x['cause'] for x in output_data] + [x['effect'] for x in output_data])
        abcd = [
            (string, sum(zipf_frequency(word, "en") for word in string.split()) / len(string.split())) if len(string) else (None, 20)
            for string in cur_vars
        ]
        print([x for x in abcd if x[1] < 2 and x[0] and len(x[0]) <= 6])


        #print(sorted(set([x['cause'] for x in output_data] + [x['effect'] for x in output_data])))

        outname = fname.replace(file_suffix, file_suffix_out)
        print(cur_vars)
        print(output_data)
        print(100*'-')
        with open(os.path.join(PATH2SAVE, outname), "w") as outfile:
            json.dump(output_data, outfile, indent=2)


if __name__ == '__main__':
	PATH2PREDS = "path/to/preds"
	PATH2SAVE  = "path/to/save/output"
	FILE_SUFFIX = "file_suffix_of_preds"
	FILE_SUFFIX_OUT = "file_suffix_for_outputs"
	MODE = 'test'  # test or val

	run_cleaning(PATH2PREDS, FILE_SUFFIX, PATH2SAVE, FILE_SUFFIX_OUT, MODE)
  
