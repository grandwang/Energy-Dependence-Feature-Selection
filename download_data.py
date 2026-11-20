# SPDX-License-Identifier: GPL-3.0-only
"""Download from the original hosts and validate before installing locally."""
import argparse
import tempfile
import urllib.request
from pathlib import Path
from data_utils import ROOT, read_json, load_dataset

def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--data-dir', type=Path, default=ROOT/'data')
    args = parser.parse_args()
    args.data_dir.mkdir(parents=True, exist_ok=True)
    for item in read_json(ROOT/'datasets.json'):
        target = args.data_dir/item['file']
        if target.exists():
            load_dataset(target, item)
            print('Verified existing:', item['dataset'])
            continue
        with tempfile.TemporaryDirectory(dir=args.data_dir) as temp:
            candidate = Path(temp)/item['file']
            request = urllib.request.Request(item['download_url'], headers={'User-Agent':'EDFS-minimal/0.1.0'})
            with urllib.request.urlopen(request, timeout=120) as response, candidate.open('wb') as dest:
                while True:
                    chunk = response.read(1024*1024)
                    if not chunk:
                        break
                    dest.write(chunk)
            load_dataset(candidate, item)
            candidate.replace(target)
        print('Downloaded and verified:', item['dataset'])

if __name__ == '__main__':
    main()
