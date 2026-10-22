"""Run the code examples of the README and the documentation."""

import re
import unittest
import warnings
from pathlib import Path

ROOT = Path(__file__).parent.parent


def python_code_blocks(path):
    return re.findall(r"```python\n(.*?)```", path.read_text(encoding="utf-8"), re.S)


class TestDocumentationExamples(unittest.TestCase):

    def assert_examples_run(self, path):
        blocks = python_code_blocks(path)
        self.assertTrue(blocks, f"No examples found in {path}")

        # The blocks of a file build on each other, so run them in one namespace.
        # Examples must not cause any warnings, e.g. from evaluating the expansion
        # where it does not converge.
        namespace = {}
        with warnings.catch_warnings():
            warnings.simplefilter("error")
            for i, block in enumerate(blocks):
                with self.subTest(file=path.name, block=i):
                    exec(compile(block, f"{path.name}, block {i}", "exec"), namespace)

    def test_readme(self):
        self.assert_examples_run(ROOT / "README.md")

    def test_usage(self):
        self.assert_examples_run(ROOT / "docs" / "usage.md")


if __name__ == "__main__":
    unittest.main()
