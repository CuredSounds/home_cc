"""
Unit tests for PostgreSQL + pgvector Database Management and Schema Migration.
"""

from __future__ import annotations

import os
import unittest
from unittest.mock import MagicMock, patch

from pg_vector_db import (
    DEFAULT_VECTOR_DIM,
    PGConfig,
    connect_db,
    get_db_connection,
    init_extensions,
    init_tables,
    init_vector_database,
    parse_args,
    verify_schema,
)


class TestPGVectorDB(unittest.TestCase):
    def test_pg_config_defaults_and_conninfo(self) -> None:
        cfg = PGConfig(
            host="db.example.com",
            port=5433,
            dbname="test_db",
            user="test_user",
            password="secret_password",
            vector_dim=512,
        )
        self.assertEqual(
            cfg.get_conninfo(),
            "postgresql://test_user:secret_password@db.example.com:5433/test_db",
        )
        self.assertEqual(cfg.vector_dim, 512)

    def test_pg_config_database_url_override(self) -> None:
        custom_url = "postgresql://myuser:mypass@remotehost:6543/proddb?sslmode=require"
        cfg = PGConfig(database_url=custom_url)
        self.assertEqual(cfg.get_conninfo(), custom_url)

    def test_pg_config_from_env(self) -> None:
        env_vars = {
            "POSTGRES_HOST": "127.0.0.1",
            "POSTGRES_PORT": "5435",
            "POSTGRES_DB": "env_db",
            "POSTGRES_USER": "env_user",
            "POSTGRES_PASSWORD": "env_password",
            "EMBEDDING_DIMENSION": "768",
        }
        with patch.dict(os.environ, env_vars, clear=False):
            # remove DATABASE_URL if present in test env
            with patch.dict(os.environ, {"DATABASE_URL": ""}):
                cfg = PGConfig.from_env()
                self.assertEqual(cfg.host, "127.0.0.1")
                self.assertEqual(cfg.port, 5435)
                self.assertEqual(cfg.dbname, "env_db")
                self.assertEqual(cfg.user, "env_user")
                self.assertEqual(cfg.password, "env_password")
                self.assertEqual(cfg.vector_dim, 768)

    def test_parse_args(self) -> None:
        args = parse_args(["--init", "--dim", "1536", "--host", "localhost", "--port", "5432"])
        self.assertTrue(args.init)
        self.assertEqual(args.dim, 1536)
        self.assertEqual(args.host, "localhost")
        self.assertEqual(args.port, 5432)

    @patch("pg_vector_db.psycopg.connect")
    def test_connect_db_and_context_manager(self, mock_connect: MagicMock) -> None:
        mock_conn = MagicMock()
        mock_connect.return_value = mock_conn

        cfg = PGConfig()
        with get_db_connection(cfg) as conn:
            self.assertEqual(conn, mock_conn)

        mock_connect.assert_called_once_with(cfg.get_conninfo(), autocommit=False)
        mock_conn.commit.assert_called_once()
        mock_conn.close.assert_called_once()

    @patch("pg_vector_db.psycopg.connect")
    def test_init_tables_and_extensions(self, mock_connect: MagicMock) -> None:
        mock_conn = MagicMock()
        mock_cursor = MagicMock()
        mock_conn.cursor.return_value.__enter__.return_value = mock_cursor
        mock_connect.return_value = mock_conn

        exts = init_extensions(mock_conn)
        self.assertIn("vector", exts)
        self.assertIn("pg_trgm", exts)

        init_tables(mock_conn, vector_dim=384, force=True)
        # Verify execute was called multiple times for drop, tables, indexes
        self.assertTrue(mock_cursor.execute.call_count >= 5)

    @patch("pg_vector_db.psycopg.connect")
    def test_verify_schema(self, mock_connect: MagicMock) -> None:
        mock_conn = MagicMock()
        mock_cursor = MagicMock()
        mock_conn.cursor.return_value.__enter__.return_value = mock_cursor

        # Mock responses for extensions, table existence, counts, columns, and indexes
        mock_cursor.fetchall.side_effect = [
            [{"extname": "vector", "extversion": "0.5.0"}, {"extname": "pg_trgm", "extversion": "1.6"}],  # extensions
            [{"column_name": "id", "udt_name": "int4", "data_type": "integer"}, {"column_name": "metadata_embedding", "udt_name": "vector", "data_type": "USER-DEFINED"}],  # books cols
            [{"column_name": "id", "udt_name": "int8", "data_type": "bigint"}, {"column_name": "embedding", "udt_name": "vector", "data_type": "USER-DEFINED"}],  # chunks cols
            [{"indexname": "idx_books_meta_embedding", "tablename": "books", "indexdef": "CREATE INDEX ..."}],  # indexes
        ]
        mock_cursor.fetchone.side_effect = [
            {"exists": True},  # books exists
            {"count": 756},    # books count
            {"exists": True},  # chunks exists
            {"count": 12000},  # chunks count
        ]

        schema = verify_schema(mock_conn)
        self.assertEqual(schema["extensions"]["vector"], "0.5.0")
        self.assertEqual(schema["tables"]["books"]["row_count"], 756)
        self.assertEqual(schema["tables"]["book_chunks"]["row_count"], 12000)
        self.assertEqual(len(schema["indexes"]), 1)


if __name__ == "__main__":
    unittest.main()
