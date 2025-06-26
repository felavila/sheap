import sys
import yaml
from collections import defaultdict

def load_line_names(path):
    """Load all line_name values from a YAML file, regardless of whether
    it uses `region:` or `common:`/`less_common:`."""
    with open(path) as f:
        data = yaml.safe_load(f)

    names = set()
    if 'region' in data:
        entries = data['region']
    else:
        entries = data.get('common', []) + data.get('less_common', [])

    for e in entries:
        name = e.get('line_name')
        if name:
            names.add(name)
    return names

def main(files):
    line_to_files = defaultdict(list)
    for fname in files:
        names = load_line_names(fname)
        for name in names:
            line_to_files[name].append(fname)

    # Find duplicates
    duplicates = {name: flist for name, flist in line_to_files.items() if len(flist) > 1}
    if not duplicates:
        print("No line_name appears in more than one file.")
    else:
        print("The following line_name(s) appear in multiple files:\n")
        for name, flist in duplicates.items():
            print(f"  • {name}: {', '.join(flist)}")

if __name__ == '__main__':
    if len(sys.argv) < 4:
        print("Usage: python3 check_duplicates.py file1.yaml file2.yaml file3.yaml")
        sys.exit(1)
    main(sys.argv[1:])
