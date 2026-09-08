-- Simplified TPC-H-style schema. Swap for your real (sampled/anonymized)
-- schema when you move past the demo. Deliberately small (6 tables) so
-- the SQLite demo runs fast; scale up row counts in generate_data.py to
-- make the index-selection problem non-trivial.
--
-- Table order matters here: parents must be created before children
-- that reference them via FOREIGN KEY. SQLite doesn't enforce FKs by
-- default so this went unnoticed there, but Postgres does enforce
-- creation order -- nation before customer/supplier, orders before
-- lineitem, etc.

CREATE TABLE nation (
    n_nationkey   INTEGER PRIMARY KEY,
    n_name        TEXT NOT NULL,
    n_regionkey   INTEGER NOT NULL
);

CREATE TABLE customer (
    c_custkey     INTEGER PRIMARY KEY,
    c_name        TEXT NOT NULL,
    c_nationkey   INTEGER NOT NULL,
    c_phone       TEXT,
    c_acctbal     REAL,
    c_mktsegment  TEXT,
    FOREIGN KEY (c_nationkey) REFERENCES nation(n_nationkey)
);

CREATE TABLE part (
    p_partkey     INTEGER PRIMARY KEY,
    p_name        TEXT NOT NULL,
    p_brand       TEXT,
    p_category    TEXT,
    p_retailprice REAL
);

CREATE TABLE supplier (
    s_suppkey     INTEGER PRIMARY KEY,
    s_name        TEXT NOT NULL,
    s_nationkey   INTEGER NOT NULL,
    s_acctbal     REAL,
    FOREIGN KEY (s_nationkey) REFERENCES nation(n_nationkey)
);

CREATE TABLE orders (
    o_orderkey     INTEGER PRIMARY KEY,
    o_custkey      INTEGER NOT NULL,
    o_orderstatus  TEXT,
    o_totalprice   REAL,
    o_orderdate    TEXT,
    o_orderpriority TEXT,
    FOREIGN KEY (o_custkey) REFERENCES customer(c_custkey)
);

CREATE TABLE lineitem (
    l_orderkey    INTEGER NOT NULL,
    l_linenumber  INTEGER NOT NULL,
    l_partkey     INTEGER NOT NULL,
    l_quantity    REAL,
    l_extendedprice REAL,
    l_discount    REAL,
    l_shipdate    TEXT,
    l_returnflag  TEXT,
    PRIMARY KEY (l_orderkey, l_linenumber),
    FOREIGN KEY (l_orderkey) REFERENCES orders(o_orderkey),
    FOREIGN KEY (l_partkey) REFERENCES part(p_partkey)
);
