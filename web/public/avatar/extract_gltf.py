#!/usr/bin/env python3
# extract_gltf_json.py
# Usage: python3 extract_gltf_json.py <input-file> <output.json>

import sys
import re
import base64
from collections import deque
from json import loads, JSONDecodeError

def find_json_by_braces(data_bytes, start_pos):
    # find first '{' at or after start_pos and extract balanced JSON by counting braces
    n = len(data_bytes)
    for i in range(start_pos, n):
        if data_bytes[i] == 0x7b:  # '{'
            depth = 0
            j = i
            while j < n:
                b = data_bytes[j]
                if b == 0x7b: depth += 1
                elif b == 0x7d: depth -= 1
                j += 1
                if depth == 0:
                    try:
                        candidate = data_bytes[i:j].decode('utf-8', errors='strict')
                    except Exception:
                        candidate = None
                    if candidate:
                        # quick sanity: must contain "asset" and "scenes" or "nodes" or "meshes"
                        if '"asset"' in candidate and ('"scenes"' in candidate or '"nodes"' in candidate or '"meshes"' in candidate):
                            try:
                                # validate JSON
                                loads(candidate)
                                return candidate
                            except JSONDecodeError:
                                pass
                    break
    return None

def try_search_raw_bytes(path):
    with open(path, 'rb') as f:
        data = f.read()
    # 1) If file is a GLB (binary glTF), header contains "glTF" at start
    if data[:4] == b'glTF' or b'glTF' in data[:64]:
        # Search for JSON chunk by locating first occurrence of b'{"asset"'
        idx = data.find(b'{"asset"')
        if idx != -1:
            found = find_json_by_braces(data, idx)
            if found:
                return found
    # 2) Generic binary/text search for JSON substring
    idx = data.find(b'{"asset"')
    if idx != -1:
        found = find_json_by_braces(data, idx)
        if found:
            return found
    # 3) Search for any '{' that leads to a glTF-like JSON (looking for "asset")
    idxs = [m.start() for m in re.finditer(b'\\{', data)]
    for idx in idxs:
        # try only positions with subsequent text containing "asset"
        window = data[idx: idx+200]
        if b'"asset"' in window:
            found = find_json_by_braces(data, idx)
            if found:
                return found
    return None

def try_base64_decode_and_search(path):
    with open(path, 'rb') as f:
        text = f.read()
    # heuristics: if file appears mostly printable and base64-like
    printable_ratio = sum(1 for b in text if 32 <= b < 127) / max(1, len(text))
    if printable_ratio < 0.7:
        return None
    # Strip whitespace/newlines and try base64 decode
    s = re.sub(b'\\s+', b'', text)
    try:
        decoded = base64.b64decode(s, validate=True)
    except Exception:
        return None
    # search decoded bytes
    idx = decoded.find(b'{"asset"')
    if idx != -1:
        found = find_json_by_braces(decoded, idx)
        if found:
            return found
    return None

def main():
    if len(sys.argv) < 3:
        print("Usage: python3 extract_gltf_json.py <input-file> <output.json>")
        sys.exit(2)
    path = sys.argv[1]
    out = sys.argv[2]

    print("Attempting raw byte search...")
    result = try_search_raw_bytes(path)
    if result:
        with open(out, 'w', encoding='utf-8') as f:
            f.write(result)
        print("SUCCESS: extracted JSON written to", out)
        return

    print("Raw search failed. Attempting base64 decode + search...")
    result = try_base64_decode_and_search(path)
    if result:
        with open(out, 'w', encoding='utf-8') as f:
            f.write(result)
        print("SUCCESS: extracted JSON written to", out)
        return

    print("No glTF JSON found by heuristics.")
    # Extra hint: check if file is a standalone .bin (buffer) or needs .gltf wrapper
    with open(path, 'rb') as f:
        header = f.read(64)
    if b'glTF' not in header:
        print("HINT: file does not contain 'glTF' header in first 64 bytes; it may be a .bin buffer or a base64 blob. If you exported separate *.gltf + *.bin, re-upload the .gltf JSON file. If you have a .glb, try using a GLB-aware tool (blender or gltf-pipeline) to convert.")
    else:
        print("File looks like a GLB but script couldn't reliably extract JSON. Provide the file again or open in Blender/importer.")

if __name__ == '__main__':
    main()
