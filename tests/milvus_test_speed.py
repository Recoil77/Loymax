import time
import random
import numpy as np
from pymilvus import connections, Collection, FieldSchema, CollectionSchema, DataType, utility

COLLECTION_NAME = "milvus_test_perf"
EMBEDDING_DIM = 1536

def create_collection():
    connections.connect(alias="default", host="192.168.168.11", port="19530")  # поменяй на свой host/port
    if utility.has_collection(COLLECTION_NAME):
        utility.drop_collection(COLLECTION_NAME)
    fields = [
        FieldSchema(name="id", dtype=DataType.INT64, is_primary=True, auto_id=False),
        FieldSchema(name="embedding", dtype=DataType.FLOAT_VECTOR, dim=EMBEDDING_DIM)
    ]
    schema = CollectionSchema(fields, description="Speed test")
    collection = Collection(COLLECTION_NAME, schema=schema)
    return collection

def main():
    print("Создаём коллекцию...")
    collection = create_collection()

    print("Вставляем 100 векторов...")
    ids = list(range(100))
    vectors = np.random.randn(100, EMBEDDING_DIM).astype(np.float32)
    insert_data = [ids, vectors.tolist()]
    collection.insert(insert_data)
    collection.flush()

    print("Создаём индекс для embedding...")
    index_params = {
        "index_type": "IVF_FLAT",
        "metric_type": "IP",
        "params": {"nlist": 16}
    }
    collection.create_index(
        field_name="embedding",
        index_params=index_params
    )

    print("Загружаем коллекцию...")
    collection.load()

    print("Делаем 100 поисков по id...")
    times = []
    for _ in range(100):
        id_query = random.choice(ids)
        start = time.perf_counter()
        expr = f"id == {id_query}"
        results = collection.query(expr, output_fields=["id", "embedding"])
        elapsed = time.perf_counter() - start
        times.append(elapsed)
        # Для контроля можно раскомментировать:
        # print(f"id={id_query}: {results}")

    print(f"Минимальное время:  {min(times)*1000:.2f} ms")
    print(f"Среднее время:     {sum(times)/len(times)*1000:.2f} ms")
    print(f"Максимальное время:{max(times)*1000:.2f} ms")

if __name__ == "__main__":
    main()
