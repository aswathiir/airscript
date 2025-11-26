import json, os
p='./avatar.gltf'
if not os.path.exists(p):
    print("avatar.gltf not found in current folder")
    raise SystemExit(1)
g=json.load(open(p,'r',encoding='utf8'))
anims=g.get('animations',[])
print("ANIMATIONS_COUNT="+str(len(anims)))
for i,a in enumerate(anims):
    print(f"[{i}] name={a.get('name','<unnamed>')} channels={len(a.get('channels',[]))} samplers={len(a.get('samplers',[]))}")
for i,b in enumerate(g.get('buffers',[])):
    print(f"[BUFFER {i}] uri={b.get('uri')} byteLength={b.get('byteLength')}")
for i,im in enumerate(g.get('images',[])):
    print(f"[IMAGE {i}] uri={im.get('uri')}")
print("FILES_IN_FOLDER:")
print('\\n'.join(sorted(os.listdir('.'))))
