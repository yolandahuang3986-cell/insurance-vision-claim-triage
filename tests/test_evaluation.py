from evaluation.error_analysis import classify_error, count_errors
from evaluation.metrics import box_iou, precision_recall


def test_box_iou_identical_boxes():
    assert box_iou([0, 0, 1, 1], [0, 0, 1, 1]) == 1.0


def test_precision_recall():
    assert precision_recall([(True, True), (True, False), (False, True)]) == (0.5, 0.5)


def test_error_categories():
    categories = [
        classify_error("dent", None, 0.9, 0.0),
        classify_error("dent", "scratch", 0.9, 0.8),
        classify_error("dent", "dent", 0.9, 0.2),
        classify_error("dent", "dent", 0.3, 0.9),
        classify_error("dent", "dent", 0.9, 0.9, image_quality_passed=False),
    ]
    assert categories == [
        "missed_damage",
        "wrong_damage_class",
        "poor_localization",
        "low_confidence_correct",
        "bad_input_quality",
    ]
    assert count_errors(categories)["missed_damage"] == 1
