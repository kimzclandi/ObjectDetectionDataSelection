"""Regression cases for user-visible broken navigation."""

import importlib.util
import tempfile
import unittest
from pathlib import Path

spec = importlib.util.spec_from_file_location("check_docs", Path(__file__).with_name("check_docs.py"))
checker = importlib.util.module_from_spec(spec)
spec.loader.exec_module(checker)


class NavigationTests(unittest.TestCase):
    def check_link(self, target, tracked=True):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            (root / "README.md").write_text(f"# Home\n[guide]({target})\n")
            (root / "guide.md").write_text("# 安装 / Setup\n## Example\n## Example\n")
            files = {"README.md", "guide.md"} if tracked else {"README.md"}
            return checker.check_documents(root, ["README.md"], files)

    def test_encoded_chinese_heading_and_duplicate(self):
        self.assertEqual(self.check_link("guide.md#%E5%AE%89%E8%A3%85--setup")[0], [])
        self.assertEqual(self.check_link("guide.md#example-1")[0], [])

    def test_missing_heading_fails_even_when_file_exists(self):
        self.assertIn("missing heading", self.check_link("guide.md#removed-section")[0][0])

    def test_same_page_anchor_and_external_link(self):
        self.assertEqual(self.check_link("#home")[0], [])
        self.assertEqual(self.check_link("https://example.org/#not-local")[1], 0)

    def test_untracked_and_traversal_targets_fail(self):
        self.assertIn("untracked", self.check_link("guide.md", tracked=False)[0][0])
        self.assertIn("escapes", self.check_link("../outside.md")[0][0])

    def test_missing_file_fails(self):
        self.assertIn("missing or untracked", self.check_link("removed.md")[0][0])

    def test_setext_and_explicit_anchor(self):
        self.assertEqual(
            checker.heading_ids('Setup\n=====\n<a id="custom"></a>'),
            {"setup", "custom"},
        )

    def test_fenced_headings_do_not_create_anchors(self):
        self.assertEqual(checker.heading_ids("~~~md\n# Not a heading\n~~~\n# Real"), {"real"})

    def test_repeated_runs_do_not_duplicate_documents(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            (root / "README.md").write_text("# Home\n[self](#home)")
            args = (root, ["README.md", "README.md"], {"README.md"})
            self.assertEqual(checker.check_documents(*args), ([], 1, 1))
            self.assertEqual(checker.check_documents(*args), ([], 1, 1))

    def test_heading_suffix_collision(self):
        self.assertEqual(checker.heading_ids("# Test\n# Test\n# Test-1"), {"test", "test-1", "test-1-1"})


if __name__ == "__main__":
    unittest.main()
