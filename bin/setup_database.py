#!/usr/bin/env python3
"""Setup PostgreSQL database and schema for twosphere-mcp project.

Creates a new database on the existing PostgreSQL server and initializes
schema for brain tensor storage, cache management, and QEC tensor metadata.

Usage:
    python bin/setup_database.py --create-db
    python bin/setup_database.py --create-schema
    python bin/setup_database.py --drop-db  # WARNING: Destroys data!

Environment:
    DATABASE_URL: postgresql://user:pass@host:port/merge2docs_dev (existing)
    TWOSPHERE_DB: twosphere_brain (new database name)
"""

import argparse
import sys
from pathlib import Path
import psycopg2
from psycopg2.extensions import ISOLATION_LEVEL_AUTOCOMMIT


# Configuration
POSTGRES_HOST = "127.0.0.1"
POSTGRES_PORT = 5432
POSTGRES_USER = "petershaw"
POSTGRES_PASSWORD = "FruitSalid4"
ADMIN_DB = "merge2docs_dev"  # Connect to existing DB for admin operations
NEW_DB = "twosphere_brain"   # New database for twosphere-mcp


def create_database():
    """Create new twosphere_brain database."""
    print(f"🔧 Creating database: {NEW_DB}")

    # Connect to admin database
    conn = psycopg2.connect(
        host=POSTGRES_HOST,
        port=POSTGRES_PORT,
        user=POSTGRES_USER,
        password=POSTGRES_PASSWORD,
        database=ADMIN_DB
    )
    conn.set_isolation_level(ISOLATION_LEVEL_AUTOCOMMIT)

    cursor = conn.cursor()

    try:
        # Check if database exists
        cursor.execute(
            "SELECT 1 FROM pg_database WHERE datname = %s",
            (NEW_DB,)
        )
        if cursor.fetchone():
            print(f"⚠️  Database {NEW_DB} already exists!")
            response = input("Drop and recreate? (yes/no): ")
            if response.lower() != "yes":
                print("Aborted.")
                return

            cursor.execute(f"DROP DATABASE {NEW_DB}")
            print(f"🗑️  Dropped existing database: {NEW_DB}")

        # Create database
        cursor.execute(f"CREATE DATABASE {NEW_DB}")
        print(f"✅ Created database: {NEW_DB}")

        # Grant privileges
        cursor.execute(f"GRANT ALL PRIVILEGES ON DATABASE {NEW_DB} TO {POSTGRES_USER}")
        print(f"✅ Granted privileges to {POSTGRES_USER}")

    finally:
        cursor.close()
        conn.close()


def create_schema():
    """Create schema for brain tensor storage."""
    print(f"🔧 Creating schema in database: {NEW_DB}")

    # Connect to new database
    conn = psycopg2.connect(
        host=POSTGRES_HOST,
        port=POSTGRES_PORT,
        user=POSTGRES_USER,
        password=POSTGRES_PASSWORD,
        database=NEW_DB
    )

    cursor = conn.cursor()

    try:
        # Enable extensions
        cursor.execute("CREATE EXTENSION IF NOT EXISTS vector")  # pgvector for embeddings
        cursor.execute("CREATE EXTENSION IF NOT EXISTS pg_trgm")  # Fuzzy text search
        print("✅ Enabled extensions: vector, pg_trgm")

        # Brain regions table (D99 atlas, 368 regions)
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS brain_regions (
                region_id SERIAL PRIMARY KEY,
                region_name VARCHAR(100) UNIQUE NOT NULL,
                atlas VARCHAR(50) NOT NULL DEFAULT 'D99',
                hemisphere VARCHAR(10),  -- 'left', 'right', 'both'
                coordinates JSONB,       -- MNI coordinates
                parent_region VARCHAR(100),
                cortical BOOLEAN DEFAULT TRUE,
                created_at TIMESTAMP DEFAULT NOW()
            )
        """)
        print("✅ Created table: brain_regions")

        # Functors table (F_i hierarchy)
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS functors (
                functor_id SERIAL PRIMARY KEY,
                functor_name VARCHAR(50) UNIQUE NOT NULL,
                hierarchy_level INTEGER NOT NULL,  -- F0, F1, F2, etc.
                description TEXT,
                created_at TIMESTAMP DEFAULT NOW()
            )
        """)
        print("✅ Created table: functors")

        # Insert default functors
        cursor.execute("""
            INSERT INTO functors (functor_name, hierarchy_level, description)
            VALUES
                ('anatomy', 0, 'Structure - what is there'),
                ('function', 1, 'Computation - what it does'),
                ('electro', 2, 'Dynamics - how it behaves'),
                ('genetics', 3, 'Heritage - why it exists'),
                ('behavior', 4, 'Task relevance - what it means'),
                ('pathology', 5, 'Disease markers - when it fails')
            ON CONFLICT (functor_name) DO NOTHING
        """)
        print("✅ Inserted default functors (6 brain functors)")

        # Scales table
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS scales (
                scale_id SERIAL PRIMARY KEY,
                scale_name VARCHAR(50) UNIQUE NOT NULL,
                description TEXT,
                created_at TIMESTAMP DEFAULT NOW()
            )
        """)
        cursor.execute("""
            INSERT INTO scales (scale_name, description)
            VALUES
                ('column', 'Cortical column level (~100μm)'),
                ('region', 'Brain region level (~1cm)'),
                ('system', 'Brain system level (~10cm)')
            ON CONFLICT (scale_name) DO NOTHING
        """)
        print("✅ Created table: scales")

        # Tensor cells table (6 functors × 380 regions × 3 scales = 6,840 cells)
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS tensor_cells (
                cell_id SERIAL PRIMARY KEY,
                region_id INTEGER REFERENCES brain_regions(region_id),
                functor_id INTEGER REFERENCES functors(functor_id),
                scale_id INTEGER REFERENCES scales(scale_id),

                -- Cell data
                features BYTEA,              -- Serialized numpy array or torch tensor
                embedding vector(768),       -- Vector embedding for similarity search

                -- Metadata
                populated BOOLEAN DEFAULT FALSE,
                training_history JSONB,      -- Training metrics, convergence info
                syndrome_score FLOAT,        -- Cross-functor syndrome detection
                last_updated TIMESTAMP DEFAULT NOW(),

                -- Constraints
                UNIQUE(region_id, functor_id, scale_id)
            )
        """)
        print("✅ Created table: tensor_cells")

        # r-IDS connections table (radius-4 Independent Dominating Set)
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS rids_connections (
                connection_id SERIAL PRIMARY KEY,
                source_region_id INTEGER REFERENCES brain_regions(region_id),
                target_region_id INTEGER REFERENCES brain_regions(region_id),

                -- r-IDS parameters
                radius INTEGER DEFAULT 4,    -- r=4 optimal for brain LID≈4-7
                geodesic_distance FLOAT,     -- Distance on cortical surface
                connection_strength FLOAT,   -- Functional connectivity weight

                -- Metadata
                connection_type VARCHAR(50),  -- 'rids', 'corpus_callosum', 'local'
                created_at TIMESTAMP DEFAULT NOW(),

                UNIQUE(source_region_id, target_region_id)
            )
        """)
        print("✅ Created table: rids_connections")

        # Cache metadata table (LRU cache tracking)
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS cache_metadata (
                cache_id SERIAL PRIMARY KEY,
                region_id INTEGER REFERENCES brain_regions(region_id),

                -- Cache statistics
                access_count INTEGER DEFAULT 0,
                last_accessed TIMESTAMP DEFAULT NOW(),
                load_latency_ms FLOAT,

                -- LRU tracking
                in_cache BOOLEAN DEFAULT FALSE,
                evicted_at TIMESTAMP,

                UNIQUE(region_id)
            )
        """)
        print("✅ Created table: cache_metadata")

        # PRIME-DE dataset metadata
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS prime_de_subjects (
                subject_id SERIAL PRIMARY KEY,
                dataset_name VARCHAR(100),
                subject_name VARCHAR(100),

                -- File paths
                nifti_path TEXT,
                timeseries_path TEXT,       -- Cached extracted timeseries

                -- Data info
                timepoints INTEGER,
                tr FLOAT,                    -- Repetition time (seconds)

                -- Processing status
                processed BOOLEAN DEFAULT FALSE,
                connectivity_computed BOOLEAN DEFAULT FALSE,

                created_at TIMESTAMP DEFAULT NOW(),

                UNIQUE(dataset_name, subject_name)
            )
        """)
        print("✅ Created table: prime_de_subjects")

        # Functional connectivity matrices
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS connectivity_matrices (
                matrix_id SERIAL PRIMARY KEY,
                subject_id INTEGER REFERENCES prime_de_subjects(subject_id),

                -- Matrix data
                matrix_data BYTEA,           -- Serialized numpy array (368×368)
                method VARCHAR(50),          -- 'distance_correlation', 'pearson', 'fft'

                -- Metadata
                computation_time_sec FLOAT,
                created_at TIMESTAMP DEFAULT NOW()
            )
        """)
        print("✅ Created table: connectivity_matrices")

        # Indices for performance
        cursor.execute("CREATE INDEX IF NOT EXISTS idx_tensor_cells_region ON tensor_cells(region_id)")
        cursor.execute("CREATE INDEX IF NOT EXISTS idx_tensor_cells_functor ON tensor_cells(functor_id)")
        cursor.execute("CREATE INDEX IF NOT EXISTS idx_tensor_cells_populated ON tensor_cells(populated)")
        cursor.execute("CREATE INDEX IF NOT EXISTS idx_rids_source ON rids_connections(source_region_id)")
        cursor.execute("CREATE INDEX IF NOT EXISTS idx_rids_target ON rids_connections(target_region_id)")
        cursor.execute("CREATE INDEX IF NOT EXISTS idx_cache_accessed ON cache_metadata(last_accessed)")
        print("✅ Created indices for performance")

        # Commit changes
        conn.commit()
        print(f"\n✅ Schema created successfully in database: {NEW_DB}")

        # Print summary
        cursor.execute("""
            SELECT table_name
            FROM information_schema.tables
            WHERE table_schema = 'public'
            ORDER BY table_name
        """)
        tables = cursor.fetchall()
        print(f"\n📊 Created {len(tables)} tables:")
        for (table,) in tables:
            cursor.execute(f"SELECT COUNT(*) FROM {table}")
            count = cursor.fetchone()[0]
            print(f"  - {table}: {count} rows")

    except Exception as e:
        conn.rollback()
        print(f"❌ Error creating schema: {e}")
        raise
    finally:
        cursor.close()
        conn.close()


def drop_database():
    """Drop twosphere_brain database (WARNING: Destroys all data!)."""
    print(f"⚠️  WARNING: This will DROP database: {NEW_DB}")
    print("⚠️  All data will be permanently destroyed!")
    response = input("Are you absolutely sure? Type 'DELETE' to confirm: ")

    if response != "DELETE":
        print("Aborted.")
        return

    # Connect to admin database
    conn = psycopg2.connect(
        host=POSTGRES_HOST,
        port=POSTGRES_PORT,
        user=POSTGRES_USER,
        password=POSTGRES_PASSWORD,
        database=ADMIN_DB
    )
    conn.set_isolation_level(ISOLATION_LEVEL_AUTOCOMMIT)

    cursor = conn.cursor()

    try:
        # Terminate active connections
        cursor.execute(f"""
            SELECT pg_terminate_backend(pg_stat_activity.pid)
            FROM pg_stat_activity
            WHERE pg_stat_activity.datname = '{NEW_DB}'
              AND pid <> pg_backend_pid()
        """)

        # Drop database
        cursor.execute(f"DROP DATABASE IF EXISTS {NEW_DB}")
        print(f"🗑️  Dropped database: {NEW_DB}")

    finally:
        cursor.close()
        conn.close()


def main():
    parser = argparse.ArgumentParser(
        description="Setup PostgreSQL database for twosphere-mcp"
    )
    parser.add_argument(
        "--create-db",
        action="store_true",
        help="Create new database"
    )
    parser.add_argument(
        "--create-schema",
        action="store_true",
        help="Create schema (tables, indices)"
    )
    parser.add_argument(
        "--drop-db",
        action="store_true",
        help="Drop database (WARNING: Destroys data!)"
    )
    parser.add_argument(
        "--all",
        action="store_true",
        help="Create database and schema (equivalent to --create-db --create-schema)"
    )

    args = parser.parse_args()

    if not any([args.create_db, args.create_schema, args.drop_db, args.all]):
        parser.print_help()
        return

    print("🧠 twosphere-mcp Database Setup")
    print(f"PostgreSQL: {POSTGRES_HOST}:{POSTGRES_PORT}")
    print(f"User: {POSTGRES_USER}")
    print(f"New database: {NEW_DB}\n")

    try:
        if args.drop_db:
            drop_database()

        if args.create_db or args.all:
            create_database()

        if args.create_schema or args.all:
            create_schema()

        print("\n✅ Database setup complete!")
        print(f"\nConnection string:")
        print(f"  postgresql://{POSTGRES_USER}:****@{POSTGRES_HOST}:{POSTGRES_PORT}/{NEW_DB}")
        print(f"\nRedis cache:")
        print(f"  redis://localhost:6379/0")

    except Exception as e:
        print(f"\n❌ Setup failed: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
