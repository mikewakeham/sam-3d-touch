"""CPU shortlist by latent score; this does not establish geometric accuracy."""
import json
from pathlib import Path
import numpy as np


class Shortlist:
    def __init__(self, top=12, per_object=2):
        self.top = top
        self.per_object = per_object
        self.objects = {}
        self.total = 0
        self.nonidentity_preferred = 0

    def add(self, metadata, prediction, scores, references):
        self.total += 1
        best = min(scores, key=scores.get)
        gain = scores['identity']-scores[best]
        self.nonidentity_preferred += int(gain > 0)
        item = dict(metadata, best_rotation=best, identity_mse=scores['identity'],
                    best_mse=scores[best], mse_reduction=gain,
                    relative_reduction=gain/scores['identity'] if scores['identity'] else 0.,
                    scores=scores)
        group = self.objects.setdefault(metadata['object_id'], [])
        # Retain only a few latents per object in memory; no bulk bank on disk.
        if len(group) == self.per_object and gain <= group[-1][0]['mse_reduction']:
            return
        arrays = dict(prediction=prediction.copy(), target=references['identity'].copy(),
                      rotated_target=references[best].copy())
        group.append((item, arrays))
        group.sort(key=lambda entry: entry[0]['mse_reduction'], reverse=True)
        del group[self.per_object:]

    def save(self, output):
        output = Path(output)
        output.mkdir(parents=True, exist_ok=True)
        ranked = sorted([entry for group in self.objects.values() for entry in group],
                        key=lambda entry: entry[0]['mse_reduction'], reverse=True)[:self.top]
        rows = []
        for rank, (metadata, arrays) in enumerate(ranked, 1):
            filename = f'{rank:02d}_{metadata["sample_id"]}_draw{metadata["draw"]}.npz'
            np.savez_compressed(output/filename, **arrays)
            rows.append(dict(metadata, rank=rank, latent_file=filename))
        report = dict(total_predictions=self.total,
                      nonidentity_preferred=self.nonidentity_preferred,
                      selection='Largest absolute latent MSE reduction; at most two draws/views per object. Post-hoc candidates, not confirmed shape matches.',
                      examples=rows)
        (output/'candidates.json').write_text(json.dumps(report, indent=2)+'\n')
        return report
