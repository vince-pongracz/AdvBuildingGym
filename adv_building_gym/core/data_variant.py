from abc import ABC, abstractmethod


class DataVariantConsumer(ABC):
    """Interface for environments that support hot-swapping datasource CSV files."""

    @abstractmethod
    def apply_data_variant(self, variant: dict[str, str]) -> None:
        """Reload datasources whose names appear in *variant*.

        Args:
            variant: Mapping of source name -> new CSV file path.
        """
