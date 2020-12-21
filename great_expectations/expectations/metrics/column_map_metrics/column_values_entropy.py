from typing import Optional

import pandas as pd

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


class ColumnValuesEntropy(ColumnMapMetricProvider):
    condition_metric_name = "column_values.entropy.under_threshold"
    condition_value_keys = (
        "double_sided",
        "threshold",
    )
    default_kwarg_values = {"threshold": None, "method": "kde", "train":  None}

    function_metric_name = "column_values.entropy"
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

    def _pandas_kde_entropy(self, column, kernel, train):
        raise NotImplementedError

    def _pandas_prophet_entropy(self, column, train, country = "us"):
        raise NotImplementedError

    def _pandas_categorical_entropy(self, column, to_look_for = None):
        # Syntax list
        default_syntax_list = [".", ",", "*"]
        if to_look_for is not None:
            default_syntax_list.append(to_look_for)

    def _pandas_bootstrap_entropy(self, column, **bootstrap_kwargs):
        raise NotImplementedError

    @column_condition_partial(engine=PandasExecutionEngine)
    def _pandas_condition(
        cls, column, _metrics, threshold, double_sided, **kwargs
    ) -> pd.Series:
        z_score, _, _ = _metrics["column_values.z_score.map"]
        try:
            if double_sided:
                under_threshold = z_score.abs() < abs(threshold)
            else:
                under_threshold = z_score < threshold
            return under_threshold
        except TypeError:
            raise (
                TypeError("Cannot check if a string lies under a numerical threshold")
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
        if metric.metric_name == "column_values.z_score.under_threshold.condition":
            return {
                "column_values.z_score.map": MetricConfiguration(
                    "column_values.z_score.map", metric.metric_domain_kwargs
                )
            }

        if metric.metric_name == "column_values.z_score.map":
            return {
                "column.mean": MetricConfiguration(
                    "column.mean", metric.metric_domain_kwargs
                ),
                "column.standard_deviation": MetricConfiguration(
                    "column.standard_deviation", metric.metric_domain_kwargs
                ),
            }

        return super()._get_evaluation_dependencies(
            metric=metric,
            configuration=configuration,
            execution_engine=execution_engine,
            runtime_configuration=runtime_configuration,
        )
