"""Run the code examples of the README and the documentation."""

import re
import unittest
import warnings
from pathlib import Path

from multipoles import ExpansionValidityWarning

ROOT = Path(__file__).parent.parent


def python_code_blocks(path):
    return re.findall(r"```python\n(.*?)```", path.read_text(encoding="utf-8"), re.DOTALL)


class TestDocumentationExamples(unittest.TestCase):

    def assert_examples_run(self, path):
        blocks = python_code_blocks(path)
        self.assertTrue(blocks, f"No examples found in {path}")

        # The blocks of a file build on each other, so run them in one namespace.
        # Examples must not cause warnings from evaluating the expansion where it does
        # not converge, or from invalid numerical operations.
        namespace = {}
        with warnings.catch_warnings():
            warnings.simplefilter("error", ExpansionValidityWarning)
            warnings.simplefilter("error", RuntimeWarning)
            for i, block in enumerate(blocks):
                with self.subTest(file=path.name, block=i):
                    exec(compile(block, f"{path.name}, block {i}", "exec"), namespace)

    def test_readme(self):
        self.assert_examples_run(ROOT / "README.md")

    def test_usage(self):
        self.assert_examples_run(ROOT / "docs" / "usage.md")


if __name__ == "__main__":
    unittest.main()
