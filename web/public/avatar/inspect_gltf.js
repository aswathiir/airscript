// inspect_gltf.js
const fs = require('fs');
const p = './avatar.gltf'; // run next to avatar.gltf
if (!fs.existsSync(p)) { console.error('avatar.gltf not found in current folder'); process.exit(1); }
const g = JSON.parse(fs.readFileSync(p,'utf8'));
const anims = g.animations || [];
console.log('ANIMATIONS_COUNT=' + anims.length);
anims.forEach((a,i)=> console.log(`[${i}] name="${a.name||'<unnamed>'}" channels=${(a.channels||[]).length} samplers=${(a.samplers||[]).length}`));
const buffers = g.buffers||[];
buffers.forEach((b,i)=> console.log(`[BUFFER ${i}] uri=${b.uri||'<embedded>'} byteLength=${b.byteLength}`));
const imgs = g.images||[];
imgs.forEach((im,i)=> console.log(`[IMAGE ${i}] uri=${im.uri||'<none>'}`));
