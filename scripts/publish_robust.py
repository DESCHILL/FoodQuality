"""Activate only the exact bundle that passed both classifier and pipeline validation."""
import argparse
import hashlib
import json
import sys
import time
from pathlib import Path

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from foodquality8.model_data import write_json


def publish(candidate,comparison,evaluation,active):
    candidate,active=Path(candidate),Path(active)
    config=json.loads(candidate.read_text(encoding='utf8'))
    comparison=json.loads(Path(comparison).read_text(encoding='utf8'))
    evaluation=json.loads(Path(evaluation).read_text(encoding='utf8'))
    sha=lambda path:hashlib.sha256(Path(path).read_bytes()).hexdigest()
    if not comparison['passed'] or not evaluation['passed'] or evaluation['split']!='val' or not evaluation['full_clean']:
        raise ValueError('Candidate did not pass complete validation; active bundle unchanged.')
    if (evaluation['config_sha256']['candidate']!=sha(candidate)
        or evaluation['config_sha256']['baseline']!=sha(active)
        or comparison['candidate_registry_sha256']!=sha(config['quality_registry'])
        or comparison['baseline_registry_sha256']!=json.loads(active.read_text(encoding='utf8'))['quality_registry_sha256']):
        raise ValueError('Validation evidence is stale.')
    for key in ('quality_registry','sic_classifier','detector'):
        if sha(config[key])!=config[key+'_sha256']:raise ValueError('Changed resource: '+key)
    q=json.loads(Path(config['quality_registry']).read_text(encoding='utf8'))
    if sha(q['weights'])!=q['sha256']:raise ValueError('Changed checkpoint.')
    pinned=active.parent/'sic'/('quality-'+sha(config['quality_registry'])[:16]+'.json')
    pinned.parent.mkdir(parents=True,exist_ok=True)
    pinned.write_bytes(Path(config['quality_registry']).read_bytes())
    config['quality_registry']=str(pinned.resolve())
    backup=active.with_name('consensus-backup-'+str(time.time_ns())+'.json')
    backup.write_bytes(active.read_bytes())
    pending=active.with_suffix('.pending.json');write_json(pending,config);pending.replace(active)
    return backup


def main():
    p=argparse.ArgumentParser()
    p.add_argument('--candidate',type=Path,required=True)
    p.add_argument('--comparison',type=Path,required=True)
    p.add_argument('--evaluation',type=Path,required=True)
    args=p.parse_args()
    print('Published; rollback:',publish(args.candidate,args.comparison,args.evaluation,ROOT/'models/consensus.json'))


if __name__=='__main__':main()
