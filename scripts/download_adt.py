"""Download one ADT sequence, verify size/SHA1, and extract only required GT files."""
import argparse
import hashlib
import json
import shutil
import urllib.request
import urllib.error
import zipfile
from pathlib import Path

DEFAULT_SEQUENCE = 'Apartment_release_golden_skeleton_seq100_10s_sample_M1292'


def download(entry, destination):
    destination = Path(destination)
    destination.parent.mkdir(parents=True, exist_ok=True)
    expected_size = int(entry['file_size_bytes'])
    expected_sha1 = entry['sha1sum']
    def valid(path):
        if not path.exists() or path.stat().st_size != expected_size:
            return False
        digest = hashlib.sha1()
        with path.open('rb') as f:
            for chunk in iter(lambda: f.read(1024*1024), b''):
                digest.update(chunk)
        return digest.hexdigest() == expected_sha1
    if valid(destination):
        print(f'Verified existing {destination.name}', flush=True)
        return
    part = destination.with_suffix(destination.suffix+'.part')
    print(f'Downloading {destination.name}: {expected_size/2**20:.1f} MiB', flush=True)
    try:
        with urllib.request.urlopen(entry['download_url'], timeout=60) as response, part.open('wb') as f:
            shutil.copyfileobj(response, f, 1024*1024)
    except urllib.error.HTTPError as e:
        raise RuntimeError(f'Download failed (HTTP {e.code}); refresh ADT download manifest if the URL expired') from None
    except urllib.error.URLError:
        raise RuntimeError('Download connection failed; check network access') from None
    if not valid(part):
        raise RuntimeError(f'Size/SHA1 verification failed for {destination.name}')
    part.replace(destination)
    print(f'SHA1 verified: {destination.name}', flush=True)


def main():
    p=argparse.ArgumentParser()
    p.add_argument('--manifest', default='data/adt_download_urls-2.json')
    p.add_argument('--sequence', default=DEFAULT_SEQUENCE)
    p.add_argument('--output', default='data/adt')
    a=p.parse_args()
    entries=json.loads(Path(a.manifest).read_text())['sequences'][a.sequence]
    root=Path(a.output)/a.sequence
    root.mkdir(parents=True, exist_ok=True)
    download(entries['main_vrs'], root/'video.vrs')
    archive=root/'main_groundtruth.zip'
    download(entries['main_groundtruth'], archive)
    required={'aria_trajectory.csv','eyegaze.csv','metadata.json'}
    extracted=set()
    with zipfile.ZipFile(archive) as z:
        for info in z.infolist():
            name=Path(info.filename).name
            if name in required:
                if name in extracted:
                    raise ValueError(f'Ambiguous duplicate file {name}')
                with z.open(info) as source, (root/name).open('wb') as target:
                    shutil.copyfileobj(source,target)
                extracted.add(name)
    if extracted != required:
        raise ValueError(f'Missing GT files {required-extracted}')
    report={'sequence':a.sequence,'groups':{k:{f:v for f,v in entries[k].items() if f!='download_url'}
            for k in ['main_vrs','main_groundtruth']},'verified_sha1':True}
    (root/'download_report.json').write_text(json.dumps(report,indent=2))
    print(root, flush=True)


if __name__=='__main__':
    main()
