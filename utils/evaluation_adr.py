import os

def read_ann_file_all_as_adr(filepath):
    """
    Read a .ann file (from meddra/) and parse all entities as ADR.
    Returns list of tuples: (label, start, end, text), with label forced to 'ADR'.
    """
    entities = []
    with open(filepath, 'r', encoding='utf-8') as file:
        for line in file:
            line = line.strip()
            if not line or line.startswith('#'):
                continue
            parts = line.split('\t')
            if len(parts) < 3:
                continue
            label_ranges = parts[1]
            entity_text = parts[2]
            label_parts = label_ranges.split(' ')
            try:
                start = int(label_parts[1])
                end = int(label_parts[2])
            except (IndexError, ValueError):
                continue
            entities.append(('ADR', start, end, entity_text))
    return entities


def read_ann_file_adr_only(filepath):
    """
    Read a .ann file (predictions) and return only ADR entities.
    """
    entities = []
    with open(filepath, 'r', encoding='utf-8') as file:
        for line in file:
            line = line.strip()
            if not line or line.startswith('#'):
                continue
            parts = line.split('\t')
            if len(parts) < 3:
                continue
            label_ranges = parts[1]
            entity_text = parts[2]
            label_parts = label_ranges.split(' ')
            label = label_parts[0]
            if label != 'ADR':
                continue
            try:
                start = int(label_parts[1])
                end = int(label_parts[2])
            except (IndexError, ValueError):
                continue
            entities.append((label, start, end, entity_text))
    return entities


def compare_entities(pred_entities, gt_entities):
    """Return sets of true positives, false positives, false negatives."""
    pred_set = set(pred_entities)
    gt_set = set(gt_entities)
    tp = pred_set & gt_set
    fp = pred_set - gt_set
    fn = gt_set - pred_set
    return tp, fp, fn


def compute_metrics(tp, fp, fn):
    """Compute precision, recall, F1."""
    precision = len(tp) / (len(tp) + len(fp)) if (len(tp) + len(fp)) > 0 else 0.0
    recall = len(tp) / (len(tp) + len(fn)) if (len(tp) + len(fn)) > 0 else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0
    return {"precision": precision, "recall": recall, "f1": f1}
