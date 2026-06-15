"""分类指标计算的单元测试。"""

from __future__ import annotations

import unittest

from src.evaluate import evaluate_predictions


class EvaluateTests(unittest.TestCase):
    """测试必要指标字段和混淆矩阵行为。"""

    def test_evaluate_predictions_reports_spam_metrics(self) -> None:
        # 构造一个简单二分类例子，检查 spam 指标和混淆矩阵字段是否稳定。
        metrics = evaluate_predictions(
            y_true=[0, 0, 1, 1],
            y_pred=[0, 1, 1, 0],
            model_name="example",
            split_name="validation",
        )

        self.assertEqual(metrics["model_name"], "example")
        self.assertEqual(metrics["split_name"], "validation")
        self.assertEqual(metrics["accuracy"], 0.5)
        self.assertEqual(metrics["spam_precision"], 0.5)
        self.assertEqual(metrics["spam_recall"], 0.5)
        self.assertEqual(metrics["spam_f1"], 0.5)
        self.assertEqual(metrics["confusion_matrix"]["labels"], ["ham", "spam"])
        self.assertEqual(metrics["confusion_matrix"]["matrix"], [[1, 1], [1, 1]])


if __name__ == "__main__":
    unittest.main()
