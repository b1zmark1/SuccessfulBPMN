from pathlib import Path

from tools.bpmn_dataset.build_dataset import (
    _map_bpmn_tag_to_class,
    _filter_pool_lane_overlaps,
    parse_bpmn_labels,
    _split_groups,
)


def test_map_bpmn_tag_to_class():
    assert _map_bpmn_tag_to_class('{http://www.omg.org/spec/BPMN/20100524/MODEL}startEvent') == 'start_event'
    assert _map_bpmn_tag_to_class('{http://www.omg.org/spec/BPMN/20100524/MODEL}exclusiveGateway') == 'gateway_exclusive'
    assert _map_bpmn_tag_to_class('{http://www.omg.org/spec/BPMN/20100524/MODEL}task') == 'task'


def test_parse_bpmn_labels_has_shapes():
    sample = Path('DATASET/bpmn-for-research-master/BPMN for Research/English/01-Dispatch-of-goods/03-Solution/Dispatch-of-goods.bpmn')
    labels = parse_bpmn_labels(sample)
    assert len(labels) > 0
    for lbl in labels:
        x, y, w, h = lbl.bbox
        assert w > 0 and h > 0


def test_split_groups_deterministic():
    paths = [Path(f'a/{i}/file{i}.bpmn') for i in range(10)]
    split1 = _split_groups(paths, seed=123)
    split2 = _split_groups(paths, seed=123)
    assert split1['train'] == split2['train']
    assert split1['val'] == split2['val']
    assert split1['test'] == split2['test']


def test_filter_pool_lane_overlaps_drops_single_lane():
    from tools.bpmn_dataset.build_dataset import ShapeLabel

    pool = ShapeLabel('pool', (0, 0, 100, 50))
    lane_same = ShapeLabel('lane', (0, 0, 100, 50))
    lane_small = ShapeLabel('lane', (0, 0, 100, 25))
    shapes = [pool, lane_same, lane_small]
    filtered = _filter_pool_lane_overlaps(shapes)
    assert pool in filtered
    assert lane_small in filtered
    assert lane_same not in filtered
