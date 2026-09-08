import json
import sqlite3
import tempfile
import unittest
from pathlib import Path

from compare_books import (
    BookRecord,
    clean_title,
    compare_sources,
    export_reports,
    generate_report,
    load_directory_titles,
    load_list_titles,
    main,
    normalize_title,
    strip_all_extensions,
    sync_to_database,
)


class TestCompareBooks(unittest.TestCase):
    def test_strip_all_extensions(self):
        self.assertEqual(strip_all_extensions("book.pdf"), "book")
        self.assertEqual(strip_all_extensions("sensor.pdf.pdf"), "sensor")
        self.assertEqual(strip_all_extensions("archive.csv.docx"), "archive")
        self.assertEqual(strip_all_extensions("simple_title"), "simple_title")

    def test_clean_title(self):
        self.assertEqual(
            clean_title("📂 Project Manifest.pdf"),
            "Project Manifest",
        )
        self.assertEqual(
            clean_title("150survivalsecrets.pdf"),
            "150survivalsecrets",
        )
        self.assertEqual(
            clean_title("- Pay Bill.pdf"),
            "Pay Bill",
        )

    def test_normalize_title(self):
        self.assertEqual(
            normalize_title("15mathconceptseverydatascientistshouldknow (1).pdf"),
            "15mathconceptseverydatascientistshouldknow",
        )
        self.assertEqual(
            normalize_title("15mathconceptseverydatascientistshouldknow.pdf"),
            "15mathconceptseverydatascientistshouldknow",
        )
        self.assertEqual(
            normalize_title("9781806029570 2.pdf"),
            "9781806029570",
        )
        self.assertEqual(
            normalize_title("9781806029570.pdf"),
            "9781806029570",
        )
        self.assertEqual(
            normalize_title("[Frank D. Petruzella] Electric Motors and Control Systems.pdf"),
            normalize_title("Frank D Petruzella Electric Motors and Control Systems.epub"),
        )

    def test_compare_sources(self):
        list_recs = [
            BookRecord(
                clean_title="Python Crash Course",
                raw_title="Python Crash Course.pdf",
                normalized_title=normalize_title("Python Crash Course.pdf"),
            ),
            BookRecord(
                clean_title="Missing Book",
                raw_title="Missing Book.epub",
                normalized_title=normalize_title("Missing Book.epub"),
            ),
        ]

        dir_recs = [
            BookRecord(
                clean_title="Python Crash Course (1)",
                raw_title="Python Crash Course (1).pdf",
                normalized_title=normalize_title("Python Crash Course (1).pdf"),
            ),
            BookRecord(
                clean_title="Extra Book In Directory",
                raw_title="Extra Book In Directory.pdf",
                normalized_title=normalize_title("Extra Book In Directory.pdf"),
            ),
        ]

        result = compare_sources(list_recs, dir_recs)
        self.assertEqual(result.total_list_items, 2)
        self.assertEqual(result.total_dir_items, 2)
        self.assertEqual(result.matched_count, 1)
        self.assertEqual(result.missing_count, 1)
        self.assertEqual(result.unlisted_count, 1)
        self.assertEqual(result.match_rate, 50.0)
        self.assertEqual(result.matched[0].list_record.clean_title, "Python Crash Course")
        self.assertEqual(result.missing_from_dir[0].clean_title, "Missing Book")
        self.assertEqual(result.unlisted_in_dir[0].clean_title, "Extra Book In Directory")

    def test_load_from_files(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            temp_path = Path(temp_dir)
            list_file = temp_path / "_test_list.txt"
            list_file.write_text(
                "# Comment\n\nBook One.pdf\nBook Two.epub\nBook Three.mobi\n",
                encoding="utf-8",
            )

            books_dir = temp_path / "books"
            books_dir.mkdir()
            (books_dir / "Book One.pdf").touch()
            (books_dir / "Book Four Unlisted.pdf").touch()

            list_records = load_list_titles(list_file)
            dir_records = load_directory_titles(books_dir)

            self.assertEqual(len(list_records), 3)
            self.assertEqual(len(dir_records), 2)

            res = compare_sources(list_records, dir_records)
            self.assertEqual(res.matched_count, 1)
            self.assertEqual(res.missing_count, 2)
            self.assertEqual(res.unlisted_count, 1)

    def test_generate_report_and_export(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            temp_path = Path(temp_dir)
            list_recs = [
                BookRecord("Book A", "Book A.pdf", "book a"),
                BookRecord("Book B", "Book B.pdf", "book b"),
            ]
            dir_recs = [
                BookRecord("Book A", "Book A.pdf", "book a", "/path/Book A.pdf"),
                BookRecord("Book C", "Book C.pdf", "book c", "/path/Book C.pdf"),
            ]
            res = compare_sources(list_recs, dir_recs)

            report = generate_report(res, verbose=True)
            self.assertIn("Matched Titles (Available):       1 (50.0%)", report)
            self.assertIn("Missing from Directory:           1", report)
            self.assertIn("Unlisted Files in Directory:      1", report)

            json_out = temp_path / "report.json"
            missing_out = temp_path / "missing.txt"
            unlisted_out = temp_path / "unlisted.txt"

            exported = export_reports(
                res,
                export_json_path=json_out,
                export_missing_path=missing_out,
                export_unlisted_path=unlisted_out,
            )

            self.assertTrue(json_out.exists())
            self.assertTrue(missing_out.exists())
            self.assertTrue(unlisted_out.exists())

            data = json.loads(json_out.read_text())
            self.assertEqual(data["summary"]["matched_count"], 1)
            self.assertEqual(missing_out.read_text().strip(), "Book B")
            self.assertEqual(unlisted_out.read_text().strip(), "Book C")

    def test_sync_to_database(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            db_path = Path(temp_dir) / "test_books.db"
            list_recs = [
                BookRecord("Book A", "Book A.pdf", "book a"),
                BookRecord("Book B", "Book B.pdf", "book b"),
            ]
            dir_recs = [
                BookRecord("Book A", "Book A.pdf", "book a", "/path/Book A.pdf"),
                BookRecord("Book C", "Book C.pdf", "book c", "/path/Book C.pdf"),
            ]
            res = compare_sources(list_recs, dir_recs)

            stats = sync_to_database(res, db_path=db_path)
            self.assertEqual(stats["matched_synced"], 1)
            self.assertEqual(stats["missing_synced"], 1)
            self.assertEqual(stats["unlisted_synced"], 1)

            with sqlite3.connect(db_path) as conn:
                cursor = conn.cursor()
                cursor.execute("SELECT clean_title, in_directory FROM books ORDER BY clean_title")
                rows = cursor.fetchall()
                self.assertEqual(
                    rows,
                    [("Book A", 1), ("Book B", 0), ("Book C", 1)],
                )

    def test_cli_execution(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            temp_path = Path(temp_dir)
            list_file = temp_path / "list.txt"
            list_file.write_text("Clean Code.pdf\nMissing Design.pdf\n")

            books_dir = temp_path / "books"
            books_dir.mkdir()
            (books_dir / "Clean Code.pdf").touch()

            db_file = temp_path / "test.db"
            export_dir = temp_path / "exports"

            code = main([
                "--list", str(list_file),
                "--dir", str(books_dir),
                "--export-dir", str(export_dir),
                "--sync-db",
                "--db-path", str(db_file),
                "--verbose",
            ])

            self.assertEqual(code, 0)
            self.assertTrue((export_dir / "missing_books.txt").exists())
            self.assertTrue((export_dir / "unlisted_books.txt").exists())
            self.assertTrue(db_file.exists())


if __name__ == "__main__":
    unittest.main()
