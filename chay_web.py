"""Resolve packaged model assets and start the customer website."""
import hashlib
import json
import os
from pathlib import Path

ROOT = Path(__file__).resolve().parent


def configure():
    model_dir = ROOT / 'models'
    active = model_dir/'consensus.json'
    if active.is_file():
        existing = json.loads(active.read_text(encoding='utf8'))
        if all(Path(existing[key]).is_file() for key in ('quality_registry','sic_classifier','detector')):
            return
    config = json.loads((model_dir/'portable.json').read_text(encoding='utf8'))
    quality = config.pop('quality')
    quality['weights'] = str(model_dir/'assets/quality.pt')
    quality_path = model_dir/'quality-runtime.json'
    quality_path.write_text(json.dumps(quality, ensure_ascii=False, indent=2), encoding='utf8')
    config['quality_registry'] = str(quality_path)
    config['quality_registry_sha256'] = hashlib.sha256(quality_path.read_bytes()).hexdigest()
    for key, filename in [('sic_classifier', 'recognition.pth'), ('detector', 'detector.pt')]:
        path = model_dir/'assets'/filename
        if hashlib.sha256(path.read_bytes()).hexdigest() != config[key+'_sha256']:
            raise ValueError(f'Trọng số bị thay đổi: {filename}')
        config[key] = str(path)
    (model_dir/'consensus.json').write_text(json.dumps(config, ensure_ascii=False, indent=2), encoding='utf8')


if __name__ == '__main__':
    import argparse
    parser=argparse.ArgumentParser()
    parser.add_argument('--profile',choices=['stable','dual'],default='stable')
    parser.add_argument('--port',type=int)
    args=parser.parse_args()
    os.chdir(ROOT)
    configure()
    if args.profile=='dual':
        config=json.loads((ROOT/'models/consensus.json').read_text(encoding='utf8'))
        config.pop('sic_classifier',None)
        config.pop('sic_classifier_sha256',None)
        config.update(mode='two_models',quality_threshold=.8)
        (ROOT/'models/dual.json').write_text(json.dumps(config,ensure_ascii=False,indent=2),encoding='utf8')
    os.environ['FOODQUALITY_PROFILE']=args.profile
    os.environ['FOODQUALITY_PORT']=str(args.port or (18502 if args.profile=='dual' else 18501))
    from launch_user_app import main
    main()
