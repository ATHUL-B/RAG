import pytest

from rag.domain import Document, Metadata, Page
from rag.errors import IngestionError
from rag.ingestion.enrichers import StaticMetadataEnricher


def _doc(doc_id: str = "3M_2018_10K", **metadata: str | int) -> Document:
    return Document(
        doc_id=doc_id, source_uri="u", pages=(Page(page_index=0, text="t"),), metadata=metadata
    )


CATALOG: dict[str, Metadata] = {"3M_2018_10K": {"company": "3M", "year": 2018}}


def test_merges_catalog_metadata_without_mutating_input() -> None:
    original = _doc(parser="pypdf-7")
    enriched = StaticMetadataEnricher(CATALOG).enrich(original)
    assert enriched.metadata == {"parser": "pypdf-7", "company": "3M", "year": 2018}
    assert original.metadata == {"parser": "pypdf-7"}


def test_unknown_document_is_an_error() -> None:
    with pytest.raises(IngestionError, match="no catalog metadata"):
        StaticMetadataEnricher(CATALOG).enrich(_doc("OTHER_2018_10K"))


def test_conflicting_value_is_an_error() -> None:
    with pytest.raises(IngestionError, match=r"conflict.*year"):
        StaticMetadataEnricher(CATALOG).enrich(_doc(year=2017))


def test_identical_existing_value_is_allowed() -> None:
    assert StaticMetadataEnricher(CATALOG).enrich(_doc(year=2018)).metadata["year"] == 2018
