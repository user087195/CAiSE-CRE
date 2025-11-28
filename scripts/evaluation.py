import os, json


PATH2PREDS = "output_r1/test_run/combined"
SUFFIX = '.json'
CONFIG = ['3-tuple', '4-tuple', '5-tuple']


def eval(preds_list, gts_list, human_eval=True, mapping_dict=None, include_pc=False, include_sig=False, ignore_imgs=False):
    """
    Evaluates predicted relations against ground truth relations.

    Both preds_list and gts_list are assumed to be lists of dictionaries following the structure:
      {
         "cause": ...,
         "effect": ...,
         "beta": ...,
         "p": ...,
         "model_id": ...,
         "moderator": ...   # optional
      }

    Parameters:
      preds_list: list of prediction dictionaries.
      gts_list: list of ground truth dictionaries.
      human_eval: if True, applies mapping_dict (if provided) to clean strings.
      mapping_dict: dictionary for mapping constructs.
      include_pc: if True, include beta in the evaluation tuples.
      include_sig: if True, include significance (p) in the evaluation tuples.
      ignore_imgs: (unused here but kept for compatibility).

    Returns:
      Tuple containing counts for correct matches, number of predictions,
      number of ground truths, precision, recall, and F1.
    """
    gt_tups = []
    for gt in gts_list:

        if ignore_imgs:
            try:
                if not gt["in_text"] and not gt["hyp_in_text"] and not gt["in_table"]:
                    continue
            except:
                print("[ERROR] Relation does not contain in_<text/table> fields:", gt)
                continue

        # Extract and normalize values from ground truth relation
        c1 = str(gt['Construct from']).lower().strip()
        c2 = str(gt['Construct to']).lower().strip()
        cm = str(gt['Construct Moderator']).lower().strip()
        si = str(gt['significance']).replace(' ', '').lower().strip()
        pc = gt['Path coefficient']

        # Optionally, remap the constructs if mapping_dict provided
        if mapping_dict and human_eval:
            if c1 in mapping_dict:
                c1 = mapping_dict[c1].strip()
            if c2 in mapping_dict:
                c2 = mapping_dict[c2].strip()
            if cm and cm in mapping_dict:
                cm = mapping_dict[cm].strip()

        # If needed, ensure beta is a float; if not, set to None
        if not isinstance(pc, float):
            pc = None
        else:
            pc = round(pc, 2)

        # Create tuple based on which extra info to include
        if not include_pc and not include_sig:
            gt_tups.append((c1, c2, cm))
        elif include_pc and not include_sig:
            gt_tups.append((c1, c2, cm, pc))
        elif not include_pc and include_sig:
            gt_tups.append((c1, c2, cm, si))
        elif include_pc and include_sig:
            gt_tups.append((c1, c2, cm, pc, si))

    gt_tups = set(gt_tups)
    # If there are no ground truth relations, return early.
    if len(gt_tups) == 0:
        return None, None, None, None, None, None

    pred_tups = []
    for p in preds_list:
        c1 = str(p.get('cause', '')).lower().strip()
        c2 = str(p.get('effect', '')).lower().strip()
        # Check for moderator; if absent, use empty string
        cm = str(p.get('moderator', '')).lower().strip()
        if cm == '':
            cm = 'none'
        sig = str(p.get('p', '')).replace(' ', '').lower().strip()
        pc = p.get('beta', None)

        if not isinstance(pc, float):
            pc = None
        else:
            pc = round(pc, 2)

        if not include_pc and not include_sig:
            pred_tups.append((c1, c2, cm))
        elif include_pc and not include_sig:
            pred_tups.append((c1, c2, cm, pc))
        elif not include_pc and include_sig:
            pred_tups.append((c1, c2, cm, sig))
        elif include_pc and include_sig:
            pred_tups.append((c1, c2, cm, pc, sig))

    pred_tups = set(pred_tups)
    #print("Prediction tuples:", pred_tups)

    # Extract the set of constructs (for optional further analysis)
    pred_cs = set([t[0] for t in pred_tups] + [t[1] for t in pred_tups] + [t[2] for t in pred_tups if t[2]])
    gt_cs   = set([t[0] for t in gt_tups] + [t[1] for t in gt_tups] + [t[2] for t in gt_tups if t[2]])

    correct = len(gt_tups.intersection(pred_tups))
    p_val   = correct / len(pred_tups) if len(pred_tups) > 0 else 0
    r_val   = correct / len(gt_tups) if len(gt_tups) > 0 else 0

    f1 = 2 * p_val * r_val / (p_val + r_val) if (p_val + r_val) > 0 else 0

    return correct, len(pred_tups), len(gt_tups), p_val, r_val, f1



def eval_disknet(path2preds,
                 suffix,
                 conf,
                 mapping_dict=None):

    if mapping_dict:
        human_eval = True
    else:
        human_eval = False

    mapping = {}
    PATH2GT = 'caise26_data/test'
    cs, ps, gs = [], [], []
    precs, recs, f1s = [], [], []
    pred_vars, gt_vars = {}, {}
    out_prec, out_rec, out_f1 = [], [], []
    gt_vars, pred_vars = {}, {}

    for conf in conf:
        for fname in os.listdir(os.path.join(PATH2GT, 'disknet_labels')):

            with open(os.path.join(PATH2GT, 'disknet_labels', fname)) as gt_file:
                gt = json.load(gt_file)

            gt_rels = gt['relations']

            if not os.path.exists(os.path.join(PATH2PREDS, fname.replace(f'.json', SUFFIX))):
                precs.append(0)
                recs.append(0)
                f1s.append(0)
                cs.append(0)
                ps.append(0)
                gs.append(len(gt_rels))
                continue
            else:
                with open(os.path.join(PATH2PREDS, fname.replace(f'.json', SUFFIX))) as predfile:
                    preds = json.load(predfile)
                    pred_rels = preds

                    zeroShot_mapping2_x = None
                    gt_vars[fname] = set([t['Construct from'].lower().strip() for t in gt_rels if t['Construct from']] + [t['Construct to'].lower().strip() for t in gt_rels if t['Construct to']] + [t['Construct Moderator'].lower().strip() for t in gt_rels if t['Construct Moderator']])
                    pred_vars[fname] = set([t['cause'].lower().strip() for t in pred_rels] + [t['effect'].lower().strip() for t in pred_rels] + [t['moderator'].lower().strip() for t in pred_rels if 'moderator' in t.keys() and t['moderator']] + [t['mediator'].lower().strip() for t in pred_rels if 'mediator' in t.keys() and t['mediator']])

                    if mapping_dict and fname in mapping_dict.keys():
                        paper_mapping = mapping_dict[fname]
                    else:
                        paper_mapping = None

                    if conf == '3-tuple':
                        nc, np, ng, p, r, f1 = eval(pred_rels,
                                                    gt_rels,
                                                    human_eval=human_eval,
                                                    mapping_dict=paper_mapping,
                                                    include_pc=False,
                                                    include_sig=False,
                                                    ignore_imgs=False)
                    if conf == '4-tuple':
                        nc, np, ng, p, r, f1 = eval(pred_rels,
                                                    gt_rels,
                                                    human_eval=human_eval,
                                                    mapping_dict=paper_mapping,
                                                    include_pc=True,
                                                    include_sig=False,
                                                    ignore_imgs=False)
                    if conf == '5-tuple':
                        nc, np, ng, p, r, f1 = eval(pred_rels,
                                                    gt_rels,
                                                    human_eval=human_eval,
                                                    mapping_dict=paper_mapping,
                                                    include_pc=True,
                                                    include_sig=True,
                                                    ignore_imgs=False)

                    if nc == None:
                        continue

                    precs.append(p)
                    recs.append(r)
                    f1s.append(f1)
                    cs.append(nc)
                    ps.append(np)
                    gs.append(ng)


        print(150*"=")
        print(f"RESULTS for {conf}:")
        print(pred_vars)
        print(gt_vars)


        prec, rec = sum(cs) / sum(ps), sum(cs) / sum(gs)
        print("Precision:", round(prec, 4))
        print("Recall:", round(rec, 4))
        print("F1:", round(2*prec*rec/(prec+rec), 4))
        out_prec.append(round(prec, 4))
        out_rec.append(round(rec, 4))
        out_f1.append(round(2*prec*rec/(prec+rec), 4))
        print("Precision:", round(sum(precs)/len(precs), 4))
        print("Recall:", round(sum(recs)/len(recs), 4))
        print("F1:", round(sum(f1s)/len(f1s), 4))
    print(150*"=")

    return out_rec, out_prec, out_f1


def eval_pubmed(path2preds=PATH2PREDS,
                suffix=SUFFIX,
                conf=CONFIG,
                mapping_dict=None):

    if mapping_dict:
        human_eval = True
    else:
        human_eval = False

    with open('caise26_data/filtered_pubmed_v2.json', 'r') as f:
        papers = json.load(f)


    cs, ps, gs = [], [], []
    precs, recs, f1s = [], [], []
    pred_vars, gt_vars = {}, {}
    out_prec, out_rec, out_f1 = [], [], []
    gt_vars, pred_vars = {}, {}

    for conf in conf:
        for fname in sorted(os.listdir('caise26_data/Test_LabelsPubmed')):
            if fname.count('_') > 1:
                idx = int(fname.split('_')[1])
                continue
            elif fname.count('_') == 1:
                idx = int(fname.split('_')[1].split('.')[0])
            else:
                continue

            # Load ground truth
            with open(f'caise26_data/Test_LabelsPubmed/{fname}', 'r') as f:
                extractions = json.load(f)

            gt_rels = extractions['relations']

            doi_url = extractions['source']
            paper = None

            for p in papers:
                if p['doi_url'] == doi_url:
                    paper = p
                    break

            if paper == None:
                continue

            paper_name_id = paper['downloaded_pdf'].replace('.pdf', '')
            if os.path.exists(f"{PATH2PREDS}/{paper_name_id}{SUFFIX}"):
                with open(f"{PATH2PREDS}/{paper_name_id}{SUFFIX}", 'r') as f:
                    preds = json.load(f)
                    pred_rels = preds
            else:
                print(f"[ERROR] Did not find predictions for:", paper_name_id)
                precs.append(0)
                recs.append(0)
                f1s.append(0)
                cs.append(0)
                ps.append(0)
                gs.append(len(gt_rels))
                continue

            gt_vars[fname] = set([t['Construct from'].lower().strip() for t in gt_rels if t['Construct from']] + [t['Construct to'].lower().strip() for t in gt_rels if t['Construct to']] + [t['Construct Moderator'].lower().strip() for t in gt_rels if t['Construct Moderator']])
            pred_vars[fname] = set([t['cause'].lower().strip() for t in pred_rels] + [t['effect'].lower().strip() for t in pred_rels] + [t['moderator'].lower().strip() for t in pred_rels if 'moderator' in t.keys() and t['moderator']] + [t['mediator'].lower().strip() for t in pred_rels if 'mediator' in t.keys() and t['mediator']])

            if mapping_dict and fname in mapping_dict.keys():
                paper_mapping = mapping_dict[fname]
            else:
                paper_mapping = None


            if conf == '3-tuple':
                nc, np, ng, p, r, f1 = eval(pred_rels,
                                            gt_rels,
                                            human_eval=human_eval,
                                            mapping_dict=paper_mapping,
                                            include_pc=False,
                                            include_sig=False,
                                            ignore_imgs=False)
            if conf == '4-tuple':
                nc, np, ng, p, r, f1 = eval(pred_rels,
                                            gt_rels,
                                            human_eval=human_eval,
                                            mapping_dict=paper_mapping,
                                            include_pc=True,
                                            include_sig=False,
                                            ignore_imgs=False)
            if conf == '5-tuple':
                nc, np, ng, p, r, f1 = eval(pred_rels,
                                            gt_rels,
                                            human_eval=human_eval,
                                            mapping_dict=paper_mapping,
                                            include_pc=True,
                                            include_sig=True,
                                            ignore_imgs=False)

            if nc == None:
                continue

            precs.append(p)
            recs.append(r)
            f1s.append(f1)
            cs.append(nc)
            ps.append(np)
            gs.append(ng)


        print(150*"=")
        print(f"RESULTS for {conf}:")
        print(pred_vars)
        print(gt_vars)

        prec, rec = sum(cs) / sum(ps), sum(cs) / sum(gs)
        print("Precision:", round(prec, 4))
        print("Recall:", round(rec, 4))
        print("F1:", round(2*prec*rec/(prec+rec), 4))
        out_prec.append(round(prec, 4))
        out_rec.append(round(rec, 4))
        out_f1.append(round(2*prec*rec/(prec+rec), 4))
        print("Precision:", round(sum(precs)/len(precs), 4))
        print("Recall:", round(sum(recs)/len(recs), 4))
        print("F1:", round(sum(f1s)/len(f1s), 4))

    print(150*"=")

    return out_rec, out_prec, out_f1
