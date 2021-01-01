from typing import Optional

import pandas as pd

from great_expectations.core import ExpectationConfiguration
from great_expectations.exceptions import GreatExpectationsError, InvalidExpectationConfigurationError
from great_expectations.exceptions.metric_exceptions import MetricProviderError
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
    default_kwarg_values = {"threshold": None, "method": "kde",}

    @column_condition_partial(engine=PandasExecutionEngine)
    def _pandas_condition(
        cls, column, _metrics, threshold, double_sided, **kwargs
    ) -> pd.Series:

        entropy = None

        # If we have calculated categorical entropy
        if "column_values.categorical_entropy" in _metrics:
            entropy, _, _ = _metrics["column_values.categorical_entropy"]

        # If we have computed bootstrap entropy
        elif "column_values.bootstrap_entropy" in _metrics:
            entropy, _, _ = _metrics["column_values.bootstrap_entropy"]

        # If we have computed kde entropy
        elif "column_values.kde_entropy" in _metrics:
            entropy, _, _ = _metrics["column_values.kde_entropy"]
        else:
            raise MetricProviderError("Entropy metric not computed")

        # Attempting to compute under_threshold metric
        try:
            if double_sided:
                under_threshold = entropy.abs() < abs(threshold)
            else:
                under_threshold = entropy < threshold
            return under_threshold
        except TypeError:
            raise (
                TypeError("Cannot check if a string or NoneType lies under a numerical threshold")
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
        # A default check
        if metric.metric_name == "column_values.entropy.under_threshold.condition":

            if "method" in configuration:
                # Categorical method of calculating entropy
                if configuration["method"].equals("categorical"):
                    return {
                        "column_values.categorical_entropy": MetricConfiguration(
                            "column_values.categorical_entropy", metric.metric_domain_kwargs
                        )
                }
                # Bootstrap method of calculating entropy
                elif configuration["method"].equals("bootstrap"):
                    return {
                        "column_values.bootstrap_entropy": MetricConfiguration(
                            "column_values.bootstrap_entropy", metric.metric_domain_kwargs
                        )
                    }
                else:
                    # Using default of kde
                    return {
                        "column_values.kde_entropy": MetricConfiguration(
                            "column_values.kde_entropy", metric.metric_domain_kwargs
                        )
                    }
            # Otherwise using default specified method: kde
            else:
                return {
                    "column_values.kde_entropy": MetricConfiguration(
                        "column_values.kde_entropy", metric.metric_domain_kwargs
                    )
                }

        return super()._get_evaluation_dependencies(
            metric=metric,
            configuration=configuration,
            execution_engine=execution_engine,
            runtime_configuration=runtime_configuration,
        )
