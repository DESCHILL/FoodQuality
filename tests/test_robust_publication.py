import hashlib
import json
import tempfile
import unittest
from pathlib import Path

from scripts.publish_robust import publish
from scripts.evaluate_robust_consensus import passes


class RobustPublicationTests(unittest.TestCase):
    def test_high_accuracy_cannot_hide_lower_coverage(self):
        baseline={'clean':dict(food_accuracy=.9,accepted_accuracy=.95,correct_and_accepted=.8)}
        for kind in ('rotation','dim_blur','occlusion'):baseline[kind]=dict(food_accuracy=.8)
        candidate={k:dict(v) for k,v in baseline.items()}
        candidate['clean'].update(accepted_accuracy=1.,correct_and_accepted=.2)
        self.assertFalse(passes(baseline,candidate))

    def test_only_matching_validation_bundle_can_replace_active(self):
        with tempfile.TemporaryDirectory() as temp:
            root=Path(temp)
            def save(name,value):
                path=root/name;path.write_text(json.dumps(value),encoding='utf8');return path
            sha=lambda path:hashlib.sha256(path.read_bytes()).hexdigest()
            weights=root/'weights.pt';weights.write_bytes(b'fixture weights')
            quality=save('quality.json',dict(weights=str(weights),sha256=sha(weights)))
            detector=root/'detector.pt';detector.write_bytes(b'detector')
            classifier=root/'classifier.pt';classifier.write_bytes(b'classifier')
            config=dict(quality_registry=str(quality),quality_registry_sha256=sha(quality),
                        detector=str(detector),detector_sha256=sha(detector),
                        sic_classifier=str(classifier),sic_classifier_sha256=sha(classifier))
            active=save('active.json',config);original=active.read_bytes()
            candidate=save('candidate.json',dict(config,new_recipe=True))
            comparison=save('comparison.json',dict(passed=True,candidate_registry_sha256=sha(quality),baseline_registry_sha256=sha(quality)))
            report=dict(passed=True,split='test',full_clean=True,config_sha256=dict(candidate=sha(candidate),baseline=sha(active)))
            evaluation=save('evaluation.json',report)
            with self.assertRaises(ValueError):publish(candidate,comparison,evaluation,active)
            self.assertEqual(original,active.read_bytes())
            report['split']='val';evaluation=save('evaluation.json',report)
            candidate.write_text('{}')
            with self.assertRaises(ValueError):publish(candidate,comparison,evaluation,active)
            self.assertEqual(original,active.read_bytes())
            candidate=save('candidate.json',dict(config,new_recipe=True))
            backup=publish(candidate,comparison,evaluation,active)
            self.assertEqual(backup.read_bytes(),original)
            self.assertTrue(json.loads(active.read_text())['new_recipe'])


if __name__=='__main__':unittest.main()
