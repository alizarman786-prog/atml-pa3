"""Task 2 data preparation (seed 6304).
Builds: COCO caption train (6,000 images x 3) / dev (96 x 1); annotation-derived grounding pair pools (train) and
dev (24 pairs) / test (64 pairs); VQAv2 complementary pairs train 640/256/640, dev 12/8/12, test 36/24/36
(yes/no, number, other). Train uses train2014 images, dev/test use val2014; all dev/test images are distinct
(image-disjoint across every dev and test suite). Downloads only the needed COCO images.
Usage: python prep_vlm.py --out /kaggle/working/vlm_data
"""
import os, io, re, json, zipfile, random, argparse, collections, concurrent.futures as cf
import urllib.request
from vqa_norm import normalize

ap = argparse.ArgumentParser()
ap.add_argument('--out', default='/kaggle/working/vlm_data')
ap.add_argument('--raw', default='/tmp/vlm_raw')
ap.add_argument('--no_images', action='store_true')
args = ap.parse_args()
os.makedirs(args.out, exist_ok=True); os.makedirs(args.raw, exist_ok=True)
SEED = 6304
rng = random.Random(SEED)

URLS = {
    'coco_ann': 'http://images.cocodataset.org/annotations/annotations_trainval2014.zip',
    'vqa_q_train': 'https://s3.amazonaws.com/cvmlp/vqa/mscoco/vqa/v2_Questions_Train_mscoco.zip',
    'vqa_q_val': 'https://s3.amazonaws.com/cvmlp/vqa/mscoco/vqa/v2_Questions_Val_mscoco.zip',
    'vqa_a_train': 'https://s3.amazonaws.com/cvmlp/vqa/mscoco/vqa/v2_Annotations_Train_mscoco.zip',
    'vqa_a_val': 'https://s3.amazonaws.com/cvmlp/vqa/mscoco/vqa/v2_Annotations_Val_mscoco.zip',
    'vqa_p_train': 'https://s3.amazonaws.com/cvmlp/vqa/mscoco/vqa/v2_Complementary_Pairs_Train_mscoco.zip',
    'vqa_p_val': 'https://s3.amazonaws.com/cvmlp/vqa/mscoco/vqa/v2_Complementary_Pairs_Val_mscoco.zip',
}


def fetch(key):
    path = f'{args.raw}/{key}.zip'
    if not os.path.exists(path):
        print('downloading', URLS[key], flush=True)
        urllib.request.urlretrieve(URLS[key], path)
    return zipfile.ZipFile(path)


def load_json_from_zip(z, name_part):
    names = [n for n in z.namelist() if name_part.lower() in n.lower() and n.endswith('.json')]
    assert names, (name_part, z.namelist())
    return json.load(z.open(names[0]))


# ------------------------------------------------------------------ COCO annotations
za = fetch('coco_ann')
cap = {s: load_json_from_zip(za, f'captions_{s}2014') for s in ['train', 'val']}
inst = {s: load_json_from_zip(za, f'instances_{s}2014') for s in ['train', 'val']}
cats = {c['id']: c['name'] for c in inst['train']['categories']}
img_info = {s: {im['id']: im for im in inst[s]['images']} for s in ['train', 'val']}

# ------------------------------------------------------------------ VQAv2
vq = {s: {q['question_id']: q for q in load_json_from_zip(fetch(f'vqa_q_{s}'), 'Questions')['questions']}
      for s in ['train', 'val']}
va = {s: {a['question_id']: a for a in load_json_from_zip(fetch(f'vqa_a_{s}'), 'Annotations')['annotations']}
      for s in ['train', 'val']}
vp = {s: load_json_from_zip(fetch(f'vqa_p_{s}'), 'complementary') for s in ['train', 'val']}
print('vqa pairs', {s: len(vp[s]) for s in vp}, flush=True)

from transformers import AutoTokenizer
tok = AutoTokenizer.from_pretrained('Qwen/Qwen3-0.6B')
ntok = lambda s: len(tok(s, add_special_tokens=False).input_ids)
Q_CAP, CAP_CAP, ANS_CAP = 80, 64, 16


def vqa_item(s, qid):
    q, a = vq[s][qid], va[s][qid]
    raw = [x['answer'] for x in a['answers']]
    norm = [normalize(x) for x in raw]
    cons, cnt = collections.Counter(norm).most_common(1)[0]
    target = collections.Counter([r for r, n in zip(raw, norm) if n == cons]).most_common(1)[0][0]
    return dict(question_id=qid, image_id=q['image_id'], question=q['question'], answer=target, consensus=cons,
                agreement=cnt, answers=raw, answer_type=a['answer_type'])


def eligible_pairs(s):
    out = []
    for q1, q2 in vp[s]:
        if q1 not in va[s] or q2 not in va[s]:
            continue
        a, b = vqa_item(s, q1), vqa_item(s, q2)
        if a['agreement'] < 8 or b['agreement'] < 8 or a['consensus'] == b['consensus']:
            continue
        if a['answer_type'] != b['answer_type'] or a['image_id'] == b['image_id']:
            continue
        if ntok(a['question']) > Q_CAP or ntok(a['answer']) > ANS_CAP or ntok(b['answer']) > ANS_CAP:
            continue
        out.append(dict(pair_id=f'{s}_{q1}_{q2}', answer_type=a['answer_type'], items=[a, b]))
    return out


TYPES = ['yes/no', 'number', 'other']
used_val = set()  # every val image used at most once across all dev/test suites


def take_pairs(pool, n_by_type, used=None):
    pool = pool[:]; rng.shuffle(pool)
    got = {t: [] for t in n_by_type}
    for p in pool:
        t = p['answer_type']
        if t not in got or len(got[t]) >= n_by_type[t]:
            continue
        ims = {it['image_id'] for it in p['items']}
        if used is not None and ims & used:
            continue
        got[t].append(p)
        if used is not None:
            used |= ims
    for t in n_by_type:
        assert len(got[t]) == n_by_type[t], (t, len(got[t]))
    return [p for t in n_by_type for p in got[t]]


vqa_train = take_pairs(eligible_pairs('train'), dict(zip(TYPES, [640, 256, 640])))
val_pool = eligible_pairs('val')
vqa_test = take_pairs(val_pool, dict(zip(TYPES, [36, 24, 36])), used_val)
vqa_dev = take_pairs(val_pool, dict(zip(TYPES, [12, 8, 12])), used_val)

# ------------------------------------------------------------------ grounding (presence / counting)
PLURAL = {'person': 'people', 'mouse': 'mice', 'sheep': 'sheep', 'skis': 'skis', 'scissors': 'scissors',
          'bus': 'buses', 'glass': 'glasses', 'wine glass': 'wine glasses', 'knife': 'knives', 'couch': 'couches',
          'sandwich': 'sandwiches', 'toothbrush': 'toothbrushes', 'bench': 'benches', 'skateboard': 'skateboards',
          'potted plant': 'potted plants', 'hair drier': 'hair driers', 'tie': 'ties', 'bus': 'buses',
          'cup': 'cups', 'dining table': 'dining tables', 'teddy bear': 'teddy bears', 'sports ball': 'sports balls'}
plural = lambda n: PLURAL.get(n, n + ('es' if n.endswith(('s', 'sh', 'ch', 'x')) else 's'))
article = lambda n: ('an ' if n[0] in 'aeiou' else 'a ') + n


def img_stats(s):
    st = collections.defaultdict(lambda: dict(present=set(), counts=collections.Counter(), bad=set(), big=set()))
    for a in inst[s]['annotations']:
        im = img_info[s][a['image_id']]; area_frac = a['area'] / (im['width'] * im['height'])
        d = st[a['image_id']]; c = a['category_id']
        d['present'].add(c)
        if a.get('iscrowd', 0) or area_frac < 0.005:
            d['bad'].add(c)  # crowd or tiny instance -> ambiguous count
        else:
            d['counts'][c] += 1
        if area_frac >= 0.01 and not a.get('iscrowd', 0):
            d['big'].add(c)
    for iid in img_info[s]:
        st[iid]
    return st


def make_grounding_pairs(s, images, n_presence, n_count, used=None):
    st = img_stats(s)
    images = [i for i in images if used is None or i not in used]
    by_pos = collections.defaultdict(list); by_cnt = collections.defaultdict(list)
    for i in images:
        for c in st[i]['big']:
            by_pos[c].append(i)
        for c, n in st[i]['counts'].items():
            if c not in st[i]['bad'] and 1 <= n <= 4:
                by_cnt[(c, n)].append(i)
    for v in list(by_pos.values()) + list(by_cnt.values()):
        rng.shuffle(v)
    cat_ids = sorted(cats)
    pres, cnt, local_used = [], [], set()

    def free(i):
        return used is None or (i not in used and i not in local_used)

    tries = 0
    while len(pres) < n_presence and tries < 200000:
        tries += 1
        c = rng.choice(cat_ids)
        pos = [i for i in by_pos[c] if free(i)]
        if not pos:
            continue
        p = rng.choice(pos)
        neg = rng.choice(images)
        if c in st[neg]['present'] or not free(neg) or neg == p:
            continue
        q = f'Does this image contain {article(cats[c])}?'
        pres.append(dict(pair_id=f'{s}_pres_{len(pres)}', kind='presence', category=cats[c], items=[
            dict(image_id=p, question=q, answer='yes'), dict(image_id=neg, question=q, answer='no')]))
        if used is not None:
            local_used.update([p, neg])
    tries = 0
    while len(cnt) < n_count and tries < 200000:
        tries += 1
        c = rng.choice(cat_ids)
        ns = [n for n in range(1, 5) if any(free(i) for i in by_cnt[(c, n)])]
        if len(ns) < 2:
            continue
        n1, n2 = rng.sample(ns, 2)
        i1 = rng.choice([i for i in by_cnt[(c, n1)] if free(i)])
        i2 = rng.choice([i for i in by_cnt[(c, n2)] if free(i) and i != i1] or [None])
        if i2 is None:
            continue
        q = f'How many {plural(cats[c])} can you see?'
        cnt.append(dict(pair_id=f'{s}_count_{len(cnt)}', kind='counting', category=cats[c], items=[
            dict(image_id=i1, question=q, answer=str(n1)), dict(image_id=i2, question=q, answer=str(n2))]))
        if used is not None:
            local_used.update([i1, i2])
    assert len(pres) == n_presence and len(cnt) == n_count, (len(pres), len(cnt))
    if used is not None:
        used |= local_used
    return pres + cnt


# captions train: 6,000 random train images x 3 captions
caps_by_img = collections.defaultdict(list)
for a in cap['train']['annotations']:
    caps_by_img[a['image_id']].append(a['caption'].strip())
train_imgs = sorted(i for i, c in caps_by_img.items() if len(c) >= 3)
cap_train_imgs = rng.sample(train_imgs, 6000)
cap_train = []
for i in cap_train_imgs:
    cs = [c for c in caps_by_img[i] if ntok(c) <= CAP_CAP]
    rng.shuffle(cs)
    assert len(cs) >= 3, i
    cap_train += [dict(image_id=i, caption=c) for c in cs[:3]]

# grounding train pools: drawn from the training images already in use (caption + VQA images)
train_pool_imgs = sorted(set(cap_train_imgs) | {it['image_id'] for p in vqa_train for it in p['items']})
ground_train = make_grounding_pairs('train', train_pool_imgs, 3000, 3000)
# grounding dev / test from val images not used by VQA dev/test
val_imgs = sorted(img_info['val'])
ground_test = make_grounding_pairs('val', val_imgs, 32, 32, used_val)
ground_dev = make_grounding_pairs('val', val_imgs, 12, 12, used_val)
# caption dev: 96 val images not used elsewhere, one caption each
vcaps = collections.defaultdict(list)
for a in cap['val']['annotations']:
    vcaps[a['image_id']].append(a['caption'].strip())
cand = [i for i in sorted(vcaps) if i not in used_val]
cap_dev = [dict(image_id=i, caption=rng.choice([c for c in vcaps[i] if ntok(c) <= CAP_CAP]))
           for i in rng.sample(cand, 96)]
used_val |= {c['image_id'] for c in cap_dev}

# ------------------------------------------------------------------ checks + save
dev_imgs = {c['image_id'] for c in cap_dev} | {it['image_id'] for p in ground_dev + vqa_dev for it in p['items']}
test_imgs = {it['image_id'] for p in ground_test + vqa_test for it in p['items']}
assert not dev_imgs & test_imgs
splits = dict(caption_train=cap_train, caption_dev=cap_dev, grounding_train_pool=ground_train,
              grounding_dev=ground_dev, grounding_test=ground_test, vqa_train=vqa_train, vqa_dev=vqa_dev,
              vqa_test=vqa_test)
for k, v in splits.items():
    json.dump(v, open(f'{args.out}/{k}.json', 'w'))
need = {'train': sorted(set(cap_train_imgs) | {it['image_id'] for p in vqa_train + ground_train for it in p['items']}),
        'val': sorted(dev_imgs | test_imgs)}
json.dump(need, open(f'{args.out}/image_ids.json', 'w'))
stats = {k: len(v) for k, v in splits.items()}
stats.update(n_train_images=len(need['train']), n_val_images=len(need['val']),
             vqa_eligible_val_pairs=len(val_pool),
             vqa_types={k: dict(collections.Counter(p['answer_type'] for p in splits[k])) for k in
                        ['vqa_train', 'vqa_dev', 'vqa_test']},
             grounding_kinds={k: dict(collections.Counter(p['kind'] for p in splits[k])) for k in
                              ['grounding_train_pool', 'grounding_dev', 'grounding_test']})
json.dump(stats, open(f'{args.out}/data_stats.json', 'w'), indent=1)
print(json.dumps(stats, indent=1), flush=True)

# ------------------------------------------------------------------ images
if not args.no_images:
    def dl(job):
        s, i = job
        fn = f'{args.out}/images/{s}2014/COCO_{s}2014_{i:012d}.jpg'
        if os.path.exists(fn):
            return 0
        for _ in range(4):
            try:
                urllib.request.urlretrieve(f'http://images.cocodataset.org/{s}2014/COCO_{s}2014_{i:012d}.jpg', fn)
                return 1
            except Exception:
                pass
        return -1
    for s in need:
        os.makedirs(f'{args.out}/images/{s}2014', exist_ok=True)
    jobs = [(s, i) for s in need for i in need[s]]
    with cf.ThreadPoolExecutor(48) as ex:
        res = list(ex.map(dl, jobs))
    print('images downloaded', res.count(1), 'existing', res.count(0), 'failed', res.count(-1), flush=True)
    assert res.count(-1) == 0
