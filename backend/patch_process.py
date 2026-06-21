import sys, re
path = 'c:/projeler/Facial-Image-Warping/backend/routers/process.py'
with open(path, 'r', encoding='utf-8') as f:
    text = f.read()

def repl(m):
    indent = m.group(1)
    varname = m.group(2)
    original = m.group(0)
    
    # Do not apply if there's already a getattr(lm check next
    
    patch = f"\n{indent}if getattr({varname}, 'ndim', 0) == 2 and {varname}.shape[1] >= 3:\n{indent}    {varname} = {varname}[:, :2]"
    return original + patch

# Match lines that assign to lm or lm2 or lm3 from landmarks.astype... or detect_face_landmarks
new_text = re.sub(r'^([ \t]+)(lm\d?)\s*=\s*(?:landmarks\.astype.*?(?:detect_face_landmarks|out).*?|detect_face_landmarks.*?)$', repl, text, flags=re.MULTILINE)

# prevent double patching
new_text = new_text.replace("if getattr(lm, 'ndim', 0) == 2 and lm.shape[1] >= 3:\n    lm = lm[:, :2]\nif getattr(lm, 'ndim', 0) == 2 and lm.shape[1] >= 3:\n    lm = lm[:, :2]", "if getattr(lm, 'ndim', 0) == 2 and lm.shape[1] >= 3:\n    lm = lm[:, :2]")

with open(path, 'w', encoding='utf-8') as f:
    f.write(new_text)

print('Patched process.py thoroughly!')
