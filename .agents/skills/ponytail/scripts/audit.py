import os, sys

def audit():
    print("=== PONYTAIL OVER-ENGINEERING AUDIT ===")
    violations = []
    for root, dirs, files in os.walk('.'):
        if any(d in root for d in ['.git', '.venv', '__pycache__', '.agents']): continue
        for file in files:
            if file.endswith('.py'):
                path = os.path.join(root, file)
                with open(path, 'r', encoding='utf-8', errors='ignore') as f:
                    lines = f.readlines()
                    loc = len(lines)
                    if loc > 300:
                        violations.append((path, f"High LOC: {loc} lines (consider pruning)"))
    if not violations:
        print("Clean! No excessive code bloat detected.")
    else:
        for p, v in violations:
            print(f"  [!] {p}: {v}")

if __name__ == '__main__':
    audit()
