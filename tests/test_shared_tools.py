"""用独立临时仓库检查共享工具的读取、校验、检索与 CLI；不验证知识内容。"""
import importlib.util
import json
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile
import unittest

SCRIPTS = Path(__file__).resolve().parents[1] / "scripts"
SPEC = importlib.util.spec_from_file_location("wiki_common_under_test", SCRIPTS / "wiki_common.py")
wiki = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(wiki)
SHARED_SPEC = importlib.util.spec_from_file_location("check_shared_under_test", SCRIPTS / "check_shared.py")
check_shared = importlib.util.module_from_spec(SHARED_SPEC)
SHARED_SPEC.loader.exec_module(check_shared)

VOCABULARY = {
    "page_types": [{"name": "algorithm", "label": "算法"}, {"name": "guide", "label": "指南"}],
    "tags": ["example", "rounding"],
    "alias_expansions": {"示例选择": ["algorithm-example"]},
    "source_topics": [{"name": "core", "label": "基础"}],
}
SCHEMAS = {
    "page": {
        "required": ["id", "title", "type", "tags", "aliases", "related", "sources",
                     "confidence", "evidence_scope", "updated"],
        "optional": ["version_sensitive"],
        "rules": {
            "id": {"type": "string", "pattern": "^[a-z][a-z0-9-]+$"},
            "type": {"type": "string", "vocabulary": "page_types"},
            "tags": {"type": "array", "vocabulary": "tags"},
            "aliases": {"type": "array"},
            "related": {"type": "array", "reference": "page"},
            "sources": {"type": "array", "reference": "source"},
            "confidence": {"type": "string", "enum": "confidence"},
            "evidence_scope": {"type": "string", "enum": "evidence_scope"},
            "updated": {"type": "date"},
        },
    },
    "source": {
        "required": ["id", "title", "category", "url", "version", "version_kind",
                     "retrieved_at", "read_form", "read_locator", "topic"],
        "optional": ["requires_external_verification", "excerpt_sha256"],
        "rules": {
            "id": {"type": "string", "pattern": "^[a-z][a-z0-9-]*$"},
            "category": {"type": "string", "enum": "source_category"},
            "version_kind": {"type": "string", "enum": "version_kind"},
            "read_form": {"type": "array", "enum": "read_form"},
            "topic": {"type": "string", "vocabulary": "source_topics"},
            "retrieved_at": {"type": "date"},
            "requires_external_verification": {"type": "boolean"},
            "excerpt_sha256": {"type": "string", "null": True},
        },
    },
}
SOURCE_SECTIONS = ("## 这份来源是什么\n\n示例论文。\n\n"
                   "## 已读定位与阅读形式\n\n§1，全文 HTML。\n\n"
                   "## 关键事实\n\n- 示例事实。\n\n"
                   "## 不能推出的结论\n\n- 不能外推到其他模型。\n")


class SharedToolsTests(unittest.TestCase):
    def setUp(self):
        temp = tempfile.TemporaryDirectory(prefix="dev_wiki_shared_")
        self.addCleanup(temp.cleanup)
        self.root = Path(temp.name) / "repository"
        (self.root / "wiki").mkdir(parents=True)
        (self.root / "sources/papers").mkdir(parents=True)
        (self.root / "data").mkdir()
        shutil.copytree(SCRIPTS, self.root / "scripts",
                        ignore=shutil.ignore_patterns("__pycache__"))
        for name in check_shared.SHARED:
            target = self.root / name
            if not target.exists():
                target.parent.mkdir(parents=True, exist_ok=True)
                target.write_text(f"# {target.name}\n", encoding="utf-8")
        (self.root / "data/vocabulary.yaml").write_text(
            json.dumps(VOCABULARY, ensure_ascii=False), encoding="utf-8")
        (self.root / "data/schemas.yaml").write_text(
            json.dumps(SCHEMAS, ensure_ascii=False), encoding="utf-8")
        self.outside = Path(temp.name) / "outside"
        self.outside.mkdir()
        self.source()

    def source(self, source_id="paper-example", body=None, **changes):
        meta = {
            "id": source_id, "title": "示例论文", "category": "paper",
            "url": "https://example.com/paper", "version": "v1",
            "version_kind": "paper-version", "retrieved_at": "2026-10-23",
            "read_form": ["html"], "read_locator": "§1", "topic": "core",
            "excerpt_sha256": None,
        }
        meta.update(changes)
        path = self.root / "sources/papers" / f"{source_id}.md"
        path.write_text("---\n" + json.dumps(meta, ensure_ascii=False) + "\n---\n\n"
                        + (SOURCE_SECTIONS if body is None else body), encoding="utf-8")
        return path

    def page(self, filename="example", body="# 示例\n\n正文。", **changes):
        meta = {
            "id": "algorithm-example", "title": "示例方法", "type": "algorithm",
            "tags": ["example"], "aliases": ["Example"], "related": [],
            "sources": ["paper-example"], "confidence": "source-reported",
            "evidence_scope": "mixed", "updated": "2026-10-23",
        }
        meta.update(changes)
        path = self.root / "wiki" / f"{filename}.md"
        path.write_text("---\n" + json.dumps(meta, ensure_ascii=False) + "\n---\n" + body + "\n",
                        encoding="utf-8")
        return path

    def cli(self, script, *args):
        return subprocess.run([sys.executable, str(self.root / "scripts" / script), *args],
                              cwd=self.outside, text=True, capture_output=True, check=False)

    def errors(self):
        return wiki.validate(self.root)[0]

    def warnings(self):
        return wiki.validate(self.root)[1]

    def test_valid_fixture(self):
        self.page()
        self.assertEqual(wiki.validate(self.root), ([], [], 1, 1))

    def test_exact_alias_ranks_before_partial_title_and_body(self):
        self.page("a", id="algorithm-a", title="GPTQ notes", aliases=[], body="# A\nGPTQ GPTQ GPTQ")
        self.page("z", id="algorithm-z", title="目标", aliases=["GPTQ"])
        rows = wiki.search(wiki.pages(self.root), "gPtQ")
        self.assertEqual([row[1].meta["id"] for row in rows], ["algorithm-z", "algorithm-a"])

    def test_query_uses_and_and_combines_tag_type_filters(self):
        self.page("both", id="algorithm-both", tags=["example"], body="# A\nactivation calibration")
        self.page("partial", id="algorithm-partial", tags=["example"], body="# B\nactivation")
        self.page("guide", id="guide-both", type="guide", tags=["example"], body="# C\nactivation calibration")
        self.page("other", id="algorithm-other", tags=["rounding"], body="# D\nactivation calibration")
        rows = wiki.search(wiki.pages(self.root), "activation calibration", tag="example", kind="algorithm")
        self.assertEqual([row[1].meta["id"] for row in rows], ["algorithm-both"])

    def test_alias_expansion_lists_registered_pages(self):
        self.page()
        result = self.cli("query.py", "--alias", "示例选择", "--json")
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertEqual([row["id"] for row in json.loads(result.stdout)], ["algorithm-example"])
        unknown = self.cli("query.py", "--alias", "未登记键")
        self.assertNotEqual(unknown.returncode, 0)
        self.assertIn("--alias 未登记", unknown.stderr)

    def test_profile_options_need_the_domain_registry(self):
        self.page()
        for option in ("--profile", "--feature", "--evidence"):
            with self.subTest(option=option):
                result = self.cli("query.py", option, "value")
                self.assertNotEqual(result.returncode, 0)
                self.assertIn("不支持 --profile/--feature/--evidence", result.stderr)

    def test_profile_options_filter_by_registry(self):
        self.page()
        self.page("other", id="algorithm-other", tags=["rounding"], body="# B\n正文。")
        (self.root / "data/architecture-profiles.json").write_text(json.dumps({
            "profiles": [{"id": "example-family", "knowledge_page": "algorithm-example",
                          "features": ["example"],
                          "evidence": [{"classification": "source-reported"}]}]},
            ensure_ascii=False), encoding="utf-8")
        rows = json.loads(self.cli("query.py", "--profile", "example-family", "--json").stdout)
        self.assertEqual([row["id"] for row in rows], ["algorithm-example"])
        rows = json.loads(self.cli("query.py", "--feature", "example", "--json").stdout)
        self.assertEqual([row["id"] for row in rows], ["algorithm-example"])
        rows = json.loads(self.cli("query.py", "--evidence", "source-reported", "--json").stdout)
        self.assertEqual([row["id"] for row in rows], ["algorithm-example"])
        unknown = self.cli("query.py", "--profile", "nope")
        self.assertNotEqual(unknown.returncode, 0)
        self.assertIn("--profile 未登记", unknown.stderr)

    def test_empty_results_have_json_contract(self):
        self.page()
        self.assertEqual(wiki.search(wiki.pages(self.root), "not-present"), [])
        result = self.cli("query.py", "not-present", "--json")
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertEqual(json.loads(result.stdout), [])

    def test_cli_is_independent_of_working_directory(self):
        self.page()
        result = self.cli("query.py", "Example", "--json")
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertEqual(json.loads(result.stdout)[0]["path"], "wiki/example.md")
        page = self.cli("get_page.py", "algorithm-example", "--follow-sources")
        self.assertEqual(page.returncode, 0, page.stderr)
        self.assertIn("示例事实", page.stdout)
        self.assertIn("https://example.com/paper", page.stdout)
        self.assertEqual(self.cli("validate.py").returncode, 0)

    def test_duplicate_page_id_is_rejected(self):
        self.page("first")
        self.page("second")
        self.assertTrue(any(error.startswith("P3") for error in self.errors()))
        result = self.cli("get_page.py", "algorithm-example")
        self.assertNotEqual(result.returncode, 0)
        self.assertIn("实际 2 个", result.stderr)

    def test_duplicate_source_id_across_files_is_rejected(self):
        self.source("paper-other", id="paper-example")
        errors = self.errors()
        self.assertTrue(any(error.startswith("S9") for error in errors))
        with self.assertRaisesRegex(ValueError, "重复来源 ID"):
            wiki.sources(self.root)

    def test_duplicate_json_key_is_rejected(self):
        path = self.page()
        content = path.read_text(encoding="utf-8")
        path.write_text(content.replace('"id": "algorithm-example"',
                                        '"id": "algorithm-hidden", "id": "algorithm-example"'),
                        encoding="utf-8")
        with self.assertRaisesRegex(ValueError, "重复"):
            wiki.pages(self.root)

    def test_wrong_page_type_is_a_validation_error(self):
        for value in (["algorithm"], {"type": "algorithm"}, None):
            with self.subTest(page_type=value):
                self.page(type=value)
                self.assertTrue(any(error.startswith("P4") for error in self.errors()))

    def test_unknown_tag_and_confidence_are_rejected(self):
        self.page(tags=["未登记标签"])
        self.assertTrue(any("不在词表中" in error for error in self.errors()))
        self.page(confidence="certain")
        self.assertTrue(any("取值非法" in error for error in self.errors()))

    def test_unknown_related_and_source_are_rejected(self):
        self.page(related=["algorithm-missing"], sources=["paper-missing"])
        errors = self.errors()
        self.assertTrue(any(error.startswith("P8") and "related 未登记 algorithm-missing" in error
                            for error in errors))
        self.assertTrue(any(error.startswith("S8") and "sources 未登记 paper-missing" in error
                            for error in errors))

    def test_reference_arrays_reject_wrong_container_or_elements(self):
        for field in ("tags", "aliases", "related", "sources"):
            for value in ("scalar", [7], [""], [None], [{"id": "wrong"}]):
                with self.subTest(field=field, value=value):
                    self.page(**{field: value})
                    self.assertTrue(any(f"{field} 必须" in error for error in self.errors()))

    def test_repeated_array_entry_is_rejected(self):
        for field in ("tags", "aliases", "related", "sources"):
            with self.subTest(field=field):
                self.page(**{field: ["same", "same"]})
                self.assertTrue(any(f"{field} 存在重复项" in error for error in self.errors()))

    def test_dates_require_calendar_date_and_hyphenated_format(self):
        for value in ("2026-02-30", "2026-13-01", "20261023", "2026-W43-5", None, 20261023):
            with self.subTest(updated=value):
                self.page(updated=value)
                self.assertTrue(any(error.startswith("P6") for error in self.errors()))

    def test_missing_local_link_is_rejected_but_external_and_code_are_ignored(self):
        self.page(body="# 示例\n[缺失](missing.md)\n[远端](https://example.com/no-file)\n"
                       "```markdown\n[示例](not-a-real-link.md)\n```\n")
        errors = self.errors()
        self.assertEqual(len(errors), 1, errors)
        self.assertIn("失效本地链接 missing.md", errors[0])

    def test_existing_space_and_percent_encoded_links_are_valid(self):
        (self.root / "wiki" / "notes file.txt").write_text("内容", encoding="utf-8")
        self.page(body="# 示例\n[尖括号](<notes file.txt>)\n[编码](notes%20file.txt)\n")
        self.assertEqual(self.errors(), [])

    def test_missing_percent_encoded_link_is_rejected(self):
        self.page(body="# 示例\n[缺失](missing%20file.md)\n")
        self.assertTrue(any("失效本地链接" in error for error in self.errors()))

    def test_heading_fragment_valid_same_and_other_page(self):
        self.page("target", id="algorithm-target", body="# Target\n\n## Valid Heading\n正文")
        self.page(body="# 示例\n\n## Local Heading\n[本页](#local-heading)\n[其他页](target.md#valid-heading)\n")
        self.assertEqual(self.errors(), [])

    def test_missing_heading_fragment_is_rejected(self):
        self.page("target", id="algorithm-target", body="# Target\n\n## Valid Heading\n正文")
        self.page(body="# 示例\n[本页](#missing-heading)\n[其他页](target.md#missing-heading)\n")
        errors = self.errors()
        self.assertEqual(len(errors), 2, errors)
        self.assertTrue(all("missing-heading" in error for error in errors))

    def test_source_requires_four_sections_with_non_empty_limits(self):
        self.source(body="## 这份来源是什么\n\n只有一节。\n")
        errors = self.errors()
        self.assertTrue(any("缺少二级标题" in error for error in errors))
        self.source(body=SOURCE_SECTIONS.replace("- 不能外推到其他模型。", ""))
        self.assertTrue(any("不得为空" in error for error in self.errors()))

    def test_source_field_set_and_enum_are_checked(self):
        self.source(category="unknown")
        self.assertTrue(any(error.startswith("S3") for error in self.errors()))
        self.source(dropped=None)
        self.assertTrue(any("表外字段" in error for error in self.errors()))

    def test_source_url_and_category_directory_are_checked(self):
        self.source(url="ftp://example.com/paper")
        self.assertTrue(any(error.startswith("S7") for error in self.errors()))
        (self.root / "sources/repos").mkdir()
        self.source("paper-misplaced")
        (self.root / "sources/papers/paper-misplaced.md").rename(
            self.root / "sources/repos/paper-misplaced.md")
        self.assertTrue(any(error.startswith("S1") for error in self.errors()))

    def test_source_excerpt_hash_must_match(self):
        import hashlib
        body = SOURCE_SECTIONS.replace("## 关键事实\n\n- 示例事实。",
                                       "## 关键事实\n\n<!-- excerpt-start -->\n引用片段\n<!-- excerpt-end -->")
        self.source(body=body)
        self.assertTrue(any("excerpt_sha256 与片段不符" in error for error in self.errors()))
        self.source(body=body, excerpt_sha256=hashlib.sha256("引用片段".encode()).hexdigest())
        self.assertEqual([error for error in self.errors() if error.startswith("S6")], [])

    def test_unreferenced_source_is_a_warning_not_an_error(self):
        self.source("paper-unused")
        self.assertEqual([error for error in self.errors() if error.startswith("S10")], [])
        self.assertTrue(any("无页面引用" in warning for warning in self.warnings()))

    def test_source_table_rows_are_rejected_outside_generated_index(self):
        (self.root / "sources" / "legacy.md").write_text(
            "| ID | 类型 | URL |\n| --- | --- | --- |\n| `paper-x` | paper | https://e.com |\n",
            encoding="utf-8")
        errors = self.errors()
        self.assertTrue(any("不得使用来源表格" in error for error in errors))
        (self.root / "sources" / "legacy.md").unlink()
        (self.root / "sources" / "INDEX.generated.md").write_text(
            "| `generated` | 生成物 |\n", encoding="utf-8")
        self.assertEqual([error for error in self.errors() if error.startswith("S11")], [])

    def test_index_check_detects_missing_and_stale_metadata(self):
        self.page()
        missing = self.cli("build_index.py", "--check")
        self.assertNotEqual(missing.returncode, 0)
        self.assertIn("索引缺失或过期", missing.stderr)
        built = self.cli("build_index.py")
        self.assertEqual(built.returncode, 0, built.stderr)
        self.assertEqual(self.cli("build_index.py", "--check").returncode, 0)
        index = self.root / "index.generated.md"
        before = index.read_text(encoding="utf-8")
        self.page(title="更新标题")
        stale = self.cli("build_index.py", "--check")
        self.assertNotEqual(stale.returncode, 0)
        self.assertIn("过期", stale.stderr)
        self.assertEqual(index.read_text(encoding="utf-8"), before)
        self.assertEqual(self.cli("build_index.py").returncode, 0)
        self.assertEqual(self.cli("build_index.py", "--check").returncode, 0)

    def test_shared_files_match_peer_and_report_missing(self):
        peer = Path(self.root.parent) / "peer"
        shutil.copytree(self.root, peer)
        result = self.cli("check_shared.py", "--peer", str(peer))
        self.assertEqual(result.returncode, 0, result.stderr)
        (peer / "scripts/wiki_common.py").write_text("# 改动\n", encoding="utf-8")
        changed = self.cli("check_shared.py", "--peer", str(peer))
        self.assertNotEqual(changed.returncode, 0)
        self.assertIn("scripts/wiki_common.py: 内容不一致", changed.stderr)


if __name__ == "__main__":
    unittest.main()
