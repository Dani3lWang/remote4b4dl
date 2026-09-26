"""Negative-query language and mask errors, separate from positive instance IoU."""


class NegativeQueryMetrics:
    def __init__(self):
        self.count = self.correct_noobj = self.mask_false_positive = 0

    def update(self, target_count, predicted_masks, token_status):
        if target_count:
            return
        self.count += 1
        self.correct_noobj += int(token_status == 'no_object')
        self.mask_false_positive += int(predicted_masks.any())

    def compute(self):
        return {
            'negative_queries': self.count,
            'negative_noobj_accuracy': self.correct_noobj / self.count if self.count else None,
            'negative_mask_false_positive_rate': self.mask_false_positive / self.count if self.count else None,
        }
