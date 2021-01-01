from typing import Optional

import pandas as pd
import scipy.stats as st
import numpy as np

from great_expectations.core import ExpectationConfiguration
from great_expectations.execution_engine import (
    ExecutionEngine,
    PandasExecutionEngine,
    SparkDFExecutionEngine,
)
from great_expectations.execution_engine.sqlalchemy_execution_engine import (
    SqlAlchemyExecutionEngine,
)
from great_expectations.expectations.metrics.import_manager import F, sa
from great_expectations.expectations.metrics.map_metric import (
    ColumnMapMetricProvider,
    column_condition_partial,
    column_function_partial,
)
from great_expectations.validator.validation_graph import MetricConfiguration


class ColumnValuesCategoricalEntropy(ColumnMapMetricProvider):
    condition_metric_name = "column_values.z_score.under_threshold"
    condition_value_keys = (
        "double_sided",
        "threshold",
    )
    default_kwarg_values = {"double_sided": True, "threshold": None}

    function_metric_name = "column_values.z_score"
    function_value_keys = tuple()

    @column_function_partial(engine=PandasExecutionEngine)
    def _pandas_function(self, column, _metrics, **kwargs):
        # return the z_score values
        mean = _metrics.get("column.mean")
        std_dev = _metrics.get("column.standard_deviation")
        try:
            return (column - mean) / std_dev
        except TypeError:
            raise (
                TypeError(
                    "Cannot complete Z-score calculations on a non-numerical column."
                )
            )

    @classmethod
    def _get_evaluation_dependencies(
        cls,
        metric: MetricConfiguration,
        configuration: Optional[ExpectationConfiguration] = None,
        execution_engine: Optional[ExecutionEngine] = None,
        runtime_configuration: Optional[dict] = None,
    ):
        """Returns a dictionary of given metric names and their corresponding configuration, specifying the metric
        types and their respective domains"""

        return super()._get_evaluation_dependencies(
            metric=metric,
            configuration=configuration,
            execution_engine=execution_engine,
            runtime_configuration=runtime_configuration,
        )
