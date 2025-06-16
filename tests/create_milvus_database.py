from pymilvus import FieldSchema, CollectionSchema, DataType, Collection

from pymilvus import connections

connections.connect(
    alias="default",
    host="192.168.168.11",
    port="19530"
)


fields = [
    FieldSchema(name="id", dtype=DataType.INT64, is_primary=True, auto_id=False),
    FieldSchema(name="ru_wiki_pageid", dtype=DataType.INT64),
    FieldSchema(name="text", dtype=DataType.VARCHAR, max_length=8192),
    FieldSchema(name="embedding", dtype=DataType.FLOAT_VECTOR, dim=1536)
]

schema = CollectionSchema(fields, description="Wiki Paragraphs")
collection = Collection("wiki_paragraphs", schema=schema)