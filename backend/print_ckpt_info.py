import torch, os
p = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'models', 'htr', 'checkpoint_finetuned.pth'))
print('checkpoint path:', p)
ck = torch.load(p, map_location='cpu')
ms = ck.get('model_state_dict', {})
print('model_state_dict keys containing lstm:')
for k in ms.keys():
    if 'lstm' in k:
        print(k, getattr(ms[k], 'shape', None))
print('\nSample top-level keys in checkpoint:', list(ck.keys()))
print('char_to_idx len:', len(ck.get('char_to_idx', {})))
