"""检查本库知识内容、领域事实源与数据层的一致性；共享工具行为见 test_shared_tools.py。"""
from __future__ import annotations

import json
import re
import subprocess
import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

from wiki_common import alias_pages, pages, search, sources, validate, vocabulary

PROFILE_CLASSIFICATIONS = {"source-reported", "source-confirmed", "config-confirmed",
                           "state-dict-confirmed", "forward-confirmed", "inferred", "unknown"}


def profiles():
    return json.loads((ROOT / "data" / "architecture-profiles.json").read_text(encoding="utf-8"))["profiles"]


class ArchitectureWikiContentTest(unittest.TestCase):
    def test_repository_validates_without_errors_or_warnings(self):
        errors, warnings, page_count, source_count = validate(ROOT)
        self.assertEqual(errors, [])
        self.assertEqual(warnings, [])
        self.assertEqual(page_count, 30)
        self.assertEqual(source_count, 28)

    def test_alias_keys_resolve_to_registered_pages(self):
        vocab = vocabulary(ROOT)
        known = {page.meta["id"] for page in pages(ROOT)}
        expansions = vocab["alias_expansions"]
        for key, targets in expansions.items():
            with self.subTest(alias=key):
                self.assertTrue(key.strip())
                self.assertTrue(targets)
                self.assertEqual(targets, list(dict.fromkeys(targets)), "别名目标不得重复")
                self.assertEqual(set(targets) - known, set())
                hits = [row[1].meta["id"] for row in search(pages(ROOT), ids=alias_pages(vocab, key))]
                self.assertEqual(hits, list(dict.fromkeys(targets)))

    def test_every_declared_tag_is_used_by_a_page(self):
        declared = [tag["name"] if isinstance(tag, dict) else tag for tag in vocabulary(ROOT)["tags"]]
        used = {tag for page in pages(ROOT) for tag in page.meta["tags"]}
        self.assertEqual(sorted(set(declared) - used), [])

    def test_every_topic_is_declared_and_populated(self):
        declared = {topic["name"] for topic in vocabulary(ROOT)["source_topics"]}
        used = {source.meta["topic"] for source in sources(ROOT).values()}
        self.assertEqual(used, declared)
        for topic in (ROOT / "sources" / "topics").glob("*.md"):
            self.assertIn(topic.stem, declared)

    def test_migration_audit_covers_every_source_exactly_once(self):
        audit = (ROOT / "sources" / "MIGRATION-AUDIT.md").read_text(encoding="utf-8")
        lookup = re.findall(r"^\| ([a-z][a-z0-9-]+) \| \d{4}-", audit, flags=re.M)
        self.assertEqual(len(lookup), len(set(lookup)))
        self.assertEqual(set(lookup), set(sources(ROOT)))

    def test_every_source_is_referenced_by_a_page_and_indexed(self):
        registry = set(sources(ROOT))
        referenced = {source_id for page in pages(ROOT) for source_id in page.meta["sources"]}
        self.assertEqual(registry - referenced, set())
        index = (ROOT / "sources" / "INDEX.generated.md").read_text(encoding="utf-8")
        self.assertEqual([source_id for source_id in registry if f"`{source_id}`" not in index], [])

    def test_generated_indices_are_current(self):
        result = subprocess.run([sys.executable, str(ROOT / "scripts" / "build_index.py"), "--check"],
                                cwd=ROOT, capture_output=True, text=True)
        self.assertEqual(result.returncode, 0, result.stderr)

    def test_profile_registry_matches_pages_and_sources(self):
        known_pages = {page.meta["id"] for page in pages(ROOT)}
        known_sources = set(sources(ROOT))
        seen = set()
        for profile in profiles():
            with self.subTest(profile=profile["id"]):
                self.assertRegex(profile["id"], r"^[a-z0-9]+(?:-[a-z0-9]+)+$")
                self.assertNotIn(profile["id"], seen)
                seen.add(profile["id"])
                self.assertIn(profile["knowledge_page"], known_pages)
                self.assertTrue(all(isinstance(item, str) for item in profile["model_type_hints"]))
                self.assertTrue(profile["implementation"]["repository"])
                self.assertTrue(profile["implementation"]["revision"])
                evidence = profile["evidence"]
                self.assertEqual(len(evidence), len(profile["features"]))
                self.assertEqual({item["feature"] for item in evidence}, set(profile["features"]))
                for item in evidence:
                    self.assertIn(item["source"], known_sources)
                    self.assertIn(item["classification"], PROFILE_CLASSIFICATIONS)


class QueryCliTest(unittest.TestCase):
    def run_query(self, *args: str) -> str:
        result = subprocess.run([sys.executable, str(ROOT / "scripts" / "query.py"), *args],
                                cwd=ROOT, capture_output=True, text=True)
        self.assertEqual(result.returncode, 0, result.stderr)
        return result.stdout

    def test_alias_expands_to_registered_pages(self):
        self.assertIn("component-attention", self.run_query("--alias", "gqa"))

    def test_profile_filter_resolves_knowledge_page(self):
        self.assertIn("model-mamba2", self.run_query("--profile", "mamba2-family"))

    def test_evidence_filter_uses_profile_provenance(self):
        self.assertIn("model-qwen2-vl", self.run_query("--profile", "qwen2-vl-family",
                                                      "--evidence", "source-reported"))

    def test_mixtral_profile_links_to_moe_knowledge_page(self):
        self.assertIn("paradigm-moe", self.run_query("--profile", "mixtral-8x7b-family"))

    def test_unknown_profile_is_rejected(self):
        result = subprocess.run([sys.executable, str(ROOT / "scripts" / "query.py"),
                                 "--profile", "not-a-profile"], cwd=ROOT, capture_output=True, text=True)
        self.assertNotEqual(result.returncode, 0)
        self.assertIn("--profile 未登记", result.stderr)


if __name__ == "__main__":
    unittest.main()
