from models.rerankers import Reranker
from indexing.collection_schema import CollectionSchema
from retrieval.types import PipelineResponse


class SearchPipeline:
    def __init__(
        self,
        retriever,
        reranker: Reranker = None,
        schema: CollectionSchema = None
    ):
        self.retriever = retriever
        self.reranker = reranker
        self.schema = schema

    def search(
        self,
        query: str,
        final_limit: int = 10,
        candidate_limit: int = 50,
        prefetch_limit: int = 100,
        with_reranker: bool = True
    ) -> list[PipelineResponse]:
        candidates = self.retriever.retrieve(
            query=query,
            candidate_limit=candidate_limit,
            prefetch_limit=prefetch_limit
        )

        if self.reranker is None or (not with_reranker):
            results = [
                PipelineResponse(
                    article_id=candidate.id,
                    score=float(candidate.score),
                    retrieval_score=float(candidate.score),
                    title=candidate.payload["title"],
                    body=candidate.payload["body_plain"]
                )
                for candidate in candidates
            ]
        
        if isinstance(self.reranker, Reranker) and with_reranker:
            documents = [
                "\n".join(
                    [
                        f"{field_name}: {candidate.payload[field_name]}"
                        for field_name in self.schema.payload_fields
                    ]
                )
                for candidate in candidates
            ]

            scores = self.reranker.score(query, documents)

            results = [
                PipelineResponse(
                    article_id=candidate.id,
                    score=float(score),
                    retrieval_score=float(candidate.score),
                    title=candidate.payload["title"],
                    body=candidate.payload["body_plain"]
                )
                for candidate, score in zip(candidates, scores)
            ]

            results.sort(
                key=lambda item: item.score,
                reverse=True
            )

        return results[:final_limit]