import torch, glob, os
ckpts = glob.glob(os.path.join(os.path.dirname(__file__), '..', 'models', 'htr', '*.pth'))
ckpts = [os.path.abspath(p) for p in ckpts]
for p in ckpts:
    try:
        ck = torch.load(p, map_location='cpu')
        num = len(ck.get('char_to_idx', {}))
        print(os.path.basename(p), 'epoch=', ck.get('epoch'), 'num_chars=', num)
    except Exception as e:
        print(os.path.basename(p), 'ERROR', e)
