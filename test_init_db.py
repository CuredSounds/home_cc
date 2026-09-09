"""
Unit tests for Database Initialization and Verification Tool (init_db.py).
"""

import sqlite3
import tempfile
import unittest
from pathlib import Path

from init_db import (
    INDEXES,
    SAMPLE_RECORDS,
    SCHEMA_COLUMNS,
    init_database,
    main,
    seed_sample_records,
    verify_database,
)


class TestInitDatabase(unittest.TestCase):

    def setUp(self):
        self.temp_dir = tempfile.TemporaryDirectory()
        self.temp_path = Path(self.temp_dir.name)
        self.db_path = self.temp_path / "test_books.db"

    def tearDown(self):
        self.temp_dir.cleanup()

    def test_init_database_creates_schema_and_indexes(self):
        stats = init_database(db_path=self.db_path)
        self.assertEqual(stats["status"], "initialized")
        self.assertEqual(stats["column_count"], 26)
        self.assertEqual(stats["indexes_count"], len(INDEXES))
        self.assertEqual(stats["row_count"], 0)

        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            cursor.execute("PRAGMA table_info(books)")
            cols = [row[1] for row in cursor.fetchall()]
            self.assertEqual(len(cols), 26)
            self.assertIn("clean_title", cols)
            self.assertIn("key_concepts", cols)
            self.assertIn("target_audience", cols)

            cursor.execute("PRAGMA index_list(books)")
            indexes = {row[1] for row in cursor.fetchall()}
            for idx_name, _ in INDEXES:
                self.assertIn(idx_name, indexes)

    def test_seed_sample_records(self):
        init_database(db_path=self.db_path)
        count = seed_sample_records(db_path=self.db_path)
        self.assertEqual(count, len(SAMPLE_RECORDS))

        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            cursor.execute("SELECT COUNT(*) FROM books")
            total = cursor.fetchone()[0]
            self.assertEqual(total, len(SAMPLE_RECORDS))

            cursor.execute("SELECT clean_title, domain, media_type FROM books WHERE clean_title = 'Electric Motors and Control Systems'")
            row = cursor.fetchone()
            self.assertIsNotNone(row)
            self.assertEqual(row[1], "Engineering & Applied Physics")
            self.assertEqual(row[2], "Book / Ebook")

    def test_verify_database_success(self):
        init_database(db_path=self.db_path)
        seed_sample_records(db_path=self.db_path)
        report = verify_database(db_path=self.db_path)

        self.assertTrue(report["valid"])
        self.assertEqual(report["tests"]["sqlite_integrity"], "PASSED")
        self.assertEqual(report["tests"]["column_schema"], "PASSED")
        self.assertEqual(report["tests"]["indexes"], "PASSED")
        self.assertEqual(report["tests"]["updated_at_trigger"], "PASSED")
        self.assertEqual(report["row_count"], len(SAMPLE_RECORDS))

    def test_verify_database_failure_on_missing_column(self):
        with sqlite3.connect(self.db_path) as conn:
            conn.execute("CREATE TABLE books (id INTEGER PRIMARY KEY, clean_title TEXT);")

        report = verify_database(db_path=self.db_path)
        self.assertFalse(report["valid"])
        self.assertEqual(report["tests"]["column_schema"], "FAILED")
        self.assertTrue(len(report["errors"]) > 0)

    def test_cli_execution_with_seed_and_verify(self):
        csv_path = self.temp_path / "seeded.csv"
        exit_code = main(["--db-path", str(self.db_path), "--force", "--seed", "--verify", "--export-csv", str(csv_path)])
        self.assertEqual(exit_code, 0)
        self.assertTrue(self.db_path.exists())
        self.assertTrue(csv_path.exists())


if __name__ == "__main__":
    unittest.main()
