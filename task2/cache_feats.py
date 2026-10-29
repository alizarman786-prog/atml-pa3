"""Cache SigLIP (google/siglip-base-patch16-224) patch features for all needed COCO images.
Uses the 196 final patch features (14x14 spatial order) = vision_model last_hidden_state, which already has SigLIP's
final post-layernorm applied exactly once (the pooling head is not used). Stored in FP16.
Usage: python cache_feats.py --data /kaggle/working/vlm_data --out /kaggle/working/feats
"""
import os, json, time, argparse
import torch
from PIL import Image
from transformers import SiglipVisionModel, SiglipImageProcessor

ap = argparse.ArgumentParser()
ap.add_argument('--data', required=True)
ap.add_argument('--images', default=None)
ap.add_argument('--out', required=True)
ap.add_argument('--bs', type=int, default=64)
args = ap.parse_args()
os.makedirs(args.out, exist_ok=True)
IMG = args.images or f'{args.data}/images'
NAME = 'google/siglip-base-patch16-224'
need = json.load(open(f'{args.data}/image_ids.json'))
proc = SiglipImageProcessor.from_pretrained(NAME)
model = SiglipVisionModel.from_pretrained(NAME, dtype=torch.float16).cuda().eval()
torch.cuda.reset_peak_memory_stats()
info = dict(model=NAME, processor=proc.to_dict(), n={})
t0 = time.time()
for split, ids in need.items():
    feats = torch.empty(len(ids), 196, 768, dtype=torch.float16)
    for b in range(0, len(ids), args.bs):
        chunk = ids[b:b + args.bs]
        ims = [Image.open(f'{IMG}/{split}2014/COCO_{split}2014_{i:012d}.jpg').convert('RGB') for i in chunk]
        px = proc(images=ims, return_tensors='pt')['pixel_values'].to('cuda', torch.float16)
        with torch.no_grad():
            h = model(pixel_values=px).last_hidden_state
        assert h.shape[1:] == (196, 768), h.shape
        feats[b:b + len(chunk)] = h.cpu()
    torch.save(dict(ids=ids, feats=feats), f'{args.out}/feats_{split}.pt')
    info['n'][split] = len(ids)
    print(split, len(ids), flush=True)
info['seconds'] = time.time() - t0
info['peak_gpu_mem_bytes'] = torch.cuda.max_memory_allocated()
info['bytes_on_disk'] = {s: os.path.getsize(f'{args.out}/feats_{s}.pt') for s in need}
json.dump(info, open(f'{args.out}/cache_info.json', 'w'), indent=1, default=str)
print(json.dumps({k: v for k, v in info.items() if k != 'processor'}, default=str))
