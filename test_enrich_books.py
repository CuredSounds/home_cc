"""
Unit tests for the Book & Media Metadata Enrichment Tool (enrich_books.py).
"""

import sqlite3
import tempfile
import unittest
from pathlib import Path

from enrich_books import (
    LocalRuleExtractor,
    MetadataCache,
    OnlineEnricher,
    calculate_title_similarity,
    clean_html_text,
    enrich_database,
    ensure_table_schema,
)


class TestEnrichBooks(unittest.TestCase):

    def test_clean_html_text(self):
        raw = "<p>This is a <b>great</b> book overview.<br>Contains technical details.</p>"
        cleaned = clean_html_text(raw)
        self.assertEqual(cleaned, "This is a great book overview. Contains technical details.")

    def test_calculate_title_similarity(self):
        s1 = "Electric Motors and Control Systems"
        s2 = "Electric motors and control systems"
        s3 = "Unrelated Cookbook Recipes"
        self.assertAlmostEqual(calculate_title_similarity(s1, s2), 1.0)
        self.assertEqual(calculate_title_similarity(s1, s3), 0.0)

    def test_extract_file_info(self):
        ft1, ext1 = LocalRuleExtractor.extract_file_info("book.pdf")
        self.assertEqual(ft1, "PDF")
        self.assertEqual(ext1, ".pdf")

        ft2, ext2 = LocalRuleExtractor.extract_file_info("manifest.md")
        self.assertEqual(ft2, "Markdown")
        self.assertEqual(ext2, ".md")

        ft3, ext3 = LocalRuleExtractor.extract_file_info("synth_bom.csv.docx")
        self.assertEqual(ft3, "Word (DOCX)")
        self.assertEqual(ext3, ".csv.docx")

        ft4, ext4 = LocalRuleExtractor.extract_file_info("course_pack_folder")
        self.assertEqual(ft4, "Directory / Folder")
        self.assertEqual(ext4, "")

    def test_extract_author(self):
        a1 = LocalRuleExtractor.extract_author("[Frank D. Petruzella] Electric Motors.pdf", "Electric Motors")
        self.assertEqual(a1, "Frank D. Petruzella")

        a2 = LocalRuleExtractor.extract_author("The Art of Electronics.pdf", "The Art of Electronics 3rd Edition")
        self.assertEqual(a2, "Paul Horowitz & Winfield Hill")

        a3 = LocalRuleExtractor.extract_author("Digital Fundamentals.pdf", "Digital Fundamentals Floyd 11th Edition")
        self.assertEqual(a3, "Thomas L. Floyd")

    def test_extract_edition_info(self):
        rev1, ed1 = LocalRuleExtractor.extract_edition_info("book_3rd_ed.pdf", "Book 3rd Edition")
        self.assertEqual(rev1, 3)
        self.assertEqual(ed1, "3rd Edition")

        rev2, ed2 = LocalRuleExtractor.extract_edition_info("pack-5", "20 Chemistry Books Collection Pack 5")
        self.assertEqual(rev2, 5)
        self.assertEqual(ed2, "Pack 5")

    def test_match_taxonomy(self):
        domain, subj, subfield, genre, aud, concepts, topics = LocalRuleExtractor.match_taxonomy(
            "deep_learning_llm.pdf", "Agentic AI From Zero to Expert"
        )
        self.assertEqual(domain, "Computer Science & Artificial Intelligence")
        self.assertEqual(subj, "Artificial Intelligence & Machine Learning")
        self.assertIn("ai", topics)

    def test_determine_media_classification(self):
        m1, c1 = LocalRuleExtractor.determine_media_classification("book.pdf", "Electric Motors", "PDF")
        self.assertEqual(m1, "Book / Ebook")
        self.assertEqual(c1, 0)

        m2, c2 = LocalRuleExtractor.determine_media_classification("udemy_course", "Udemy - Python Bootcamp", "Directory / Folder")
        self.assertEqual(m2, "Course / Video Series")

        m3, c3 = LocalRuleExtractor.determine_media_classification("130_math_books", "130+ Math Learning Books", "Directory / Folder")
        self.assertEqual(m3, "Collection / Resource Pack")
        self.assertEqual(c3, 1)

    def test_enrich_database_offline(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            temp_path = Path(temp_dir)
            db_path = temp_path / "test_enrich.db"
            csv_path = temp_path / "test_enrich.csv"

            with sqlite3.connect(db_path) as conn:
                conn.execute(
                    """
                    CREATE TABLE books (
                        id INTEGER PRIMARY KEY,
                        clean_title TEXT NOT NULL UNIQUE,
                        raw_title TEXT NOT NULL,
                        source_path TEXT,
                        in_directory INTEGER DEFAULT 0
                    );
                    """
                )
                conn.execute(
                    """
                    INSERT INTO books (clean_title, raw_title, source_path, in_directory)
                    VALUES
                    ('Electric Motors and Control Systems', '[Frank D. Petruzella] Electric Motors.pdf', '/path/motors.pdf', 1),
                    ('15 Math Concepts Every Data Scientist Should Know', '15mathconceptseverydatascientistshouldknow.pdf', '/path/math.pdf', 0),
                    ('Project Manifest: The Cybernetic Instrument', '📂 Project Manifest.md', '/path/manifest.md', 1);
                    """
                )

            stats = enrich_database(
                db_path=db_path,
                export_csv_path=csv_path,
                offline_only=True,
                max_workers=2,
            )

            self.assertEqual(stats["total_records"], 3)
            self.assertTrue(csv_path.exists())

            with sqlite3.connect(db_path) as conn:
                cursor = conn.cursor()
                cursor.execute("SELECT clean_title, author, domain, subject, genre, file_type, media_type FROM books ORDER BY id")
                rows = cursor.fetchall()

                self.assertEqual(rows[0][0], "Electric Motors and Control Systems")
                self.assertEqual(rows[0][1], "Frank D. Petruzella")
                self.assertEqual(rows[0][5], "PDF")
                self.assertEqual(rows[0][6], "Book / Ebook")

                self.assertEqual(rows[1][0], "15 Math Concepts Every Data Scientist Should Know")
                self.assertEqual(rows[1][2], "Computer Science & Artificial Intelligence")

                self.assertEqual(rows[2][0], "Project Manifest: The Cybernetic Instrument")
                self.assertEqual(rows[2][5], "Markdown")
                self.assertEqual(rows[2][6], "Project Documentation / Manifest")


if __name__ == "__main__":
    unittest.main()
