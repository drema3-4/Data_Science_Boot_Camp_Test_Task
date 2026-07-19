def recall_at_k(
    predicted: list[int],
    relevant: list[int],
    k: int
) -> float:
    predicted_at_k = predicted[:k]
    return sum(article_id in relevant for article_id in predicted_at_k) / len(relevant)

def recall_k(
    predicted_ids: list[list[int]],
    relevants: list[list[int]],
    k: int
) -> float:
    return sum (
        recall_at_k(
            predicted=predicted,
            relevant=relevant,
            k=k
        )
        for predicted, relevant in zip(predicted_ids, relevants)
    ) / len(predicted_ids)

def average_precision_at_k(
    predicted: list[int],
    relevant: list[int],
    k: int
) -> float:
    relevant_set = set(relevant)
    predicted_at_k = predicted[:k]

    hits = 0
    precision_sum = 0.0

    for index, article_id in enumerate(predicted_at_k, start=1):
        if article_id in relevant_set:
            hits += 1
            precision_sum += hits / index

    return precision_sum / min(len(relevant_set), k)

def map_k(
    predicted_ids: list[list[int]],
    relevants: list[list[int]],
    k: int
) -> float:
    return sum (
        average_precision_at_k(
            predicted=predicted,
            relevant=relevant,
            k=k
        )
        for predicted, relevant in zip(predicted_ids, relevants)
    ) / len(predicted_ids)