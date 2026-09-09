"""
PostgreSQL + pgvector Database Management, Schema Migration, and Connection Pooling.

Handles database connection setup, extension management (vector, pg_trgm),
hierarchical table schemas (books, book_chunks), and HNSW vector indexing.
"""

from __future__ import annotations

import argparse
import os
import sys
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Generator, List, Optional, Tuple, Union

try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass

import psycopg
from psycopg import Connection, sql
from psycopg.rows import dict_row

try:
    from pgvector.psycopg import register_vector
except ImportError:
    register_vector = None


DEFAULT_VECTOR_DIM: int = 384


@dataclass
class PGConfig:
    host: str = "localhost"
    port: int = 5432
    dbname: str = "home_cc"
    user: str = "postgres"
    password: str = "postgres"
    database_url: Optional[str] = None
    vector_dim: int = DEFAULT_VECTOR_DIM

    @classmethod
    def from_env(cls, vector_dim: Optional[int] = None) -> "PGConfig":
        """Loads configuration from environment variables."""
        db_url = os.getenv("DATABASE_URL")
        host = os.getenv("POSTGRES_HOST", "localhost")
        port_str = os.getenv("POSTGRES_PORT", "5432")
        port = int(port_str) if port_str.isdigit() else 5432
        dbname = os.getenv("POSTGRES_DB", "home_cc")
        user = os.getenv("POSTGRES_USER", "postgres")
        password = os.getenv("POSTGRES_PASSWORD", "postgres")

        dim_str = os.getenv("EMBEDDING_DIMENSION", str(DEFAULT_VECTOR_DIM))
        resolved_dim = vector_dim if vector_dim is not None else (int(dim_str) if dim_str.isdigit() else DEFAULT_VECTOR_DIM)

        return cls(
            host=host,
            port=port,
            dbname=dbname,
            user=user,
            password=password,
            database_url=db_url,
            vector_dim=resolved_dim,
        )

    def get_conninfo(self) -> str:
        """Returns the PostgreSQL connection string URI."""
        if self.database_url:
            return self.database_url
        return f"postgresql://{self.user}:{self.password}@{self.host}:{self.port}/{self.dbname}"


def connect_db(config: Optional[PGConfig] = None) -> Connection:
    """
    Establishes a connection to PostgreSQL and registers pgvector types.
    """
    cfg = config or PGConfig.from_env()
    conninfo = cfg.get_conninfo()
    conn = psycopg.connect(conninfo, autocommit=False)
    if register_vector:
        try:
            register_vector(conn)
        except Exception:
            # Extension might not be loaded yet before init
            pass
    return conn


@contextmanager
def get_db_connection(config: Optional[PGConfig] = None) -> Generator[Connection, None, None]:
    """
    Context manager providing a transactional connection to PostgreSQL.
    """
    conn = connect_db(config)
    try:
        yield conn
        conn.commit()
    except Exception:
        conn.rollback()
        raise
    finally:
        conn.close()


def init_extensions(conn: Connection) -> List[str]:
    """
    Ensures required PostgreSQL extensions (vector, pg_trgm) are loaded.
    """
    with conn.cursor() as cur:
        cur.execute("CREATE EXTENSION IF NOT EXISTS vector;")
        cur.execute("CREATE EXTENSION IF NOT EXISTS pg_trgm;")
    conn.commit()

    if register_vector:
        try:
            register_vector(conn)
        except Exception:
            pass

    return ["vector", "pg_trgm"]


def init_tables(conn: Connection, vector_dim: int = DEFAULT_VECTOR_DIM, force: bool = False) -> None:
    """
    Creates the hierarchical two-tier schema:
    1. books (catalog level with metadata vector embedding)
    2. book_chunks (passage level with granular text vector embedding and tsvector)
    """
    with conn.cursor() as cur:
        if force:
            cur.execute("DROP TABLE IF EXISTS book_chunks CASCADE;")
            cur.execute("DROP TABLE IF EXISTS books CASCADE;")

        # 1. Books Catalog Table
        cur.execute(
            sql.SQL(
                """
                CREATE TABLE IF NOT EXISTS books (
                    id SERIAL PRIMARY KEY,
                    sqlite_id INTEGER UNIQUE,
                    clean_title TEXT NOT NULL,
                    subtitle TEXT,
                    raw_title TEXT,
                    author TEXT,
                    publisher TEXT,
                    published TEXT,
                    edition TEXT,
                    language TEXT DEFAULT 'en',
                    domain TEXT,
                    subject TEXT,
                    subfield TEXT,
                    genre TEXT,
                    topics TEXT,
                    key_concepts TEXT,
                    target_audience TEXT,
                    media_type TEXT,
                    summary TEXT,
                    file_type TEXT,
                    file_extension TEXT,
                    source_path TEXT,
                    in_directory INTEGER DEFAULT 0,
                    metadata_embedding vector({dim}),
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                );
                """
            ).format(dim=sql.Literal(vector_dim))
        )

        # 2. Granular Passage Chunks Table
        cur.execute(
            sql.SQL(
                """
                CREATE TABLE IF NOT EXISTS book_chunks (
                    id BIGSERIAL PRIMARY KEY,
                    book_id INTEGER NOT NULL REFERENCES books(id) ON DELETE CASCADE,
                    chunk_index INTEGER NOT NULL,
                    page_number INTEGER,
                    section_title TEXT,
                    content TEXT NOT NULL,
                    token_count INTEGER,
                    embedding vector({dim}) NOT NULL,
                    tsv tsvector GENERATED ALWAYS AS (to_tsvector('english', content)) STORED,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                );
                """
            ).format(dim=sql.Literal(vector_dim))
        )

        # 3. Create Indexes
        # HNSW Vector indexes
        cur.execute(
            """
            CREATE INDEX IF NOT EXISTS idx_books_meta_embedding
            ON books USING hnsw (metadata_embedding vector_cosine_ops);
            """
        )
        cur.execute(
            """
            CREATE INDEX IF NOT EXISTS idx_chunks_embedding
            ON book_chunks USING hnsw (embedding vector_cosine_ops);
            """
        )

        # Full text GIN index
        cur.execute(
            """
            CREATE INDEX IF NOT EXISTS idx_chunks_tsv
            ON book_chunks USING gin (tsv);
            """
        )

        # Relational & Filtering indexes
        cur.execute("CREATE INDEX IF NOT EXISTS idx_chunks_book_id ON book_chunks (book_id);")
        cur.execute("CREATE INDEX IF NOT EXISTS idx_books_clean_title ON books (clean_title);")
        cur.execute("CREATE INDEX IF NOT EXISTS idx_books_sqlite_id ON books (sqlite_id);")
        cur.execute("CREATE INDEX IF NOT EXISTS idx_books_domain ON books (domain);")
        cur.execute("CREATE INDEX IF NOT EXISTS idx_books_subject ON books (subject);")

    conn.commit()


def init_vector_database(
    config: Optional[PGConfig] = None,
    force: bool = False,
    vector_dim: Optional[int] = None,
) -> Dict[str, Any]:
    """
    Full database initialization workflow.
    Ensures extensions, tables, and indexes are provisioned.
    """
    cfg = config or PGConfig.from_env(vector_dim=vector_dim)
    dim = vector_dim or cfg.vector_dim

    with get_db_connection(cfg) as conn:
        exts = init_extensions(conn)
        init_tables(conn, vector_dim=dim, force=force)
        status = verify_schema(conn)

    return {
        "status": "success",
        "extensions": exts,
        "vector_dim": dim,
        "schema": status,
    }


def verify_schema(conn: Connection) -> Dict[str, Any]:
    """
    Verifies database tables, installed extensions, vector dimensions, and indexes.
    """
    results: Dict[str, Any] = {
        "extensions": {},
        "tables": {},
        "indexes": [],
    }

    with conn.cursor(row_factory=dict_row) as cur:
        # Check extensions
        cur.execute("SELECT extname, extversion FROM pg_extension WHERE extname IN ('vector', 'pg_trgm');")
        for row in cur.fetchall():
            results["extensions"][row["extname"]] = row["extversion"]

        # Check table row counts and vector columns
        for tbl in ["books", "book_chunks"]:
            cur.execute(
                sql.SQL(
                    """
                    SELECT EXISTS (
                        SELECT FROM information_schema.tables 
                        WHERE table_schema = 'public' AND table_name = {tbl}
                    ) as exists;
                    """
                ).format(tbl=sql.Literal(tbl))
            )
            exists_row = cur.fetchone()
            tbl_exists = exists_row["exists"] if exists_row else False

            if tbl_exists:
                cur.execute(sql.SQL("SELECT count(*) as count FROM {};").format(sql.Identifier(tbl)))
                cnt_row = cur.fetchone()
                cnt = cnt_row["count"] if cnt_row else 0

                cur.execute(
                    sql.SQL(
                        """
                        SELECT column_name, udt_name, data_type 
                        FROM information_schema.columns 
                        WHERE table_schema = 'public' AND table_name = {tbl};
                        """
                    ).format(tbl=sql.Literal(tbl))
                )
                cols = {c["column_name"]: c["udt_name"] for c in cur.fetchall()}

                results["tables"][tbl] = {
                    "exists": True,
                    "row_count": cnt,
                    "columns": cols,
                }
            else:
                results["tables"][tbl] = {
                    "exists": False,
                    "row_count": 0,
                    "columns": {},
                }

        # Check indexes
        cur.execute(
            """
            SELECT indexname, tablename, indexdef 
            FROM pg_indexes 
            WHERE schemaname = 'public' AND tablename IN ('books', 'book_chunks')
            ORDER BY tablename, indexname;
            """
        )
        for row in cur.fetchall():
            results["indexes"].append({
                "table": row["tablename"],
                "name": row["indexname"],
                "definition": row["indexdef"],
            })

    return results


def parse_args(args: Optional[List[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="PostgreSQL + pgvector Database Schema and Provisioning Tool."
    )
    parser.add_argument(
        "--init",
        action="store_true",
        help="Initialize extensions, tables, and HNSW indexes",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Drop existing tables and recreate from scratch (use with caution)",
    )
    parser.add_argument(
        "--status",
        "--verify",
        action="store_true",
        help="Check connection and verify database schema, extensions, and tables",
    )
    parser.add_argument(
        "--dim",
        type=int,
        default=None,
        help=f"Vector embedding dimension (default: from .env or {DEFAULT_VECTOR_DIM})",
    )
    parser.add_argument(
        "--host",
        type=str,
        default=None,
        help="PostgreSQL host (overrides environment)",
    )
    parser.add_argument(
        "--port",
        type=int,
        default=None,
        help="PostgreSQL port (overrides environment)",
    )
    parser.add_argument(
        "--db",
        type=str,
        default=None,
        help="PostgreSQL database name (overrides environment)",
    )
    parser.add_argument(
        "--user",
        type=str,
        default=None,
        help="PostgreSQL user (overrides environment)",
    )
    parser.add_argument(
        "--password",
        type=str,
        default=None,
        help="PostgreSQL password (overrides environment)",
    )
    return parser.parse_args(args)


def main(args: Optional[List[str]] = None) -> int:
    parsed = parse_args(args)
    cfg = PGConfig.from_env(vector_dim=parsed.dim)

    if parsed.host:
        cfg.host = parsed.host
    if parsed.port:
        cfg.port = parsed.port
    if parsed.db:
        cfg.dbname = parsed.db
    if parsed.user:
        cfg.user = parsed.user
    if parsed.password:
        cfg.password = parsed.password

    try:
        if parsed.init or parsed.force:
            print(f"Connecting to PostgreSQL at {cfg.host}:{cfg.port}/{cfg.dbname}...")
            res = init_vector_database(config=cfg, force=parsed.force, vector_dim=cfg.vector_dim)
            print(f"Schema initialized successfully with vector dimension {res['vector_dim']}.")
            print(f"Extensions loaded: {', '.join(res['extensions'])}")
            for tbl, details in res["schema"]["tables"].items():
                print(f"  - Table '{tbl}': {details['row_count']} rows, {len(details['columns'])} columns")
            return 0

        # Default action: verify status
        print(f"Checking connection to PostgreSQL at {cfg.host}:{cfg.port}/{cfg.dbname}...")
        with get_db_connection(cfg) as conn:
            schema_info = verify_schema(conn)

        print("\n================ PostgreSQL + pgvector Status ================")
        print("Installed Extensions:")
        for ext, ver in schema_info["extensions"].items():
            print(f"  ✓ {ext} (version {ver})")
        if not schema_info["extensions"]:
            print("  [!] No vector or pg_trgm extensions found. Run with --init to install.")

        print("\nTables:")
        for tbl, details in schema_info["tables"].items():
            if details["exists"]:
                print(f"  ✓ {tbl}: {details['row_count']} row(s), {len(details['columns'])} columns")
            else:
                print(f"  [!] {tbl}: NOT CREATED")

        print(f"\nIndexes ({len(schema_info['indexes'])} active):")
        for idx in schema_info["indexes"]:
            print(f"  - {idx['table']}.{idx['name']}")
        print("==============================================================")
        return 0

    except psycopg.OperationalError as e:
        print(f"\n[!] PostgreSQL Connection Error: {e}", file=sys.stderr)
        print("    Ensure PostgreSQL is running (e.g. `docker compose up -d`) and credentials in .env are correct.", file=sys.stderr)
        return 1
    except Exception as e:
        print(f"\n[!] Error: {e}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    sys.exit(main())
