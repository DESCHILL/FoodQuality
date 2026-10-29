"""Food-conditional quality evidence with an optional validation-selected legacy head."""
import torch


def condition_logits(checkpoint,features,legacy_features=None):
    def logits(head,raw):
        x=((raw-head['mean'])/head['std']).clamp(-8,8)
        return (x@head['quality_head']['weight'].T+head['quality_head']['bias']).reshape(-1,9,2)
    current=logits(checkpoint,features)
    legacy=checkpoint.get('legacy_quality')
    if legacy is None:return current
    if legacy_features is None:raise ValueError('Legacy quality head requires its original 224-pixel features.')
    weight=float(legacy['weight'])
    if not 0<=weight<=1:raise ValueError('Invalid quality ensemble weight.')
    probability=current.softmax(-1)*(1-weight)+logits(legacy,legacy_features).softmax(-1)*weight
    return probability.clamp_min(1e-8).log()
