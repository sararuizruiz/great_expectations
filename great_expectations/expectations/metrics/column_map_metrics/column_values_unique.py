from typing import Optional

from great_expectations.core import ExpectationConfiguration
from great_expectations.execution_engine import (
    ExecutionEngine,
    PandasExecutionEngine,
    SparkDFExecutionEngine,
)
from great_expectations.execution_engine.execution_engine import (
    MetricPartialFunctionTypes,
)
from great_expectations.execution_engine.sqlalchemy_execution_engine import (
    SqlAlchemyExecutionEngine,
)
from great_expectations.expectations.metrics.import_manager import F, Window
from great_expectations.expectations.metrics.map_metric import (
    ColumnMapMetricProvider,
    column_condition_partial,
)
from great_expectations.expectations.metrics.map_metric import sa as sa
from great_expectations.validator.validation_graph import MetricConfiguration


class ColumnValuesUnique(ColumnMapMetricProvider):
    condition_metric_name = "column_values.unique"

    @column_condition_partial(engine=PandasExecutionEngine)
    def _pandas(cls, column, **kwargs):
        return ~column.duplicated(keep=False)

    @column_condition_partial(
        engine=SqlAlchemyExecutionEngine,
    )
    def _sqlalchemy(cls, column, _table, **kwargs):
        dup_query = (
            sa.select([column])
            .select_from(_table)
            .group_by(column)
            .having(sa.func.count(column) > 1)
        )

        return column.notin_(dup_query)

    @column_condition_partial(
        engine=SqlAlchemyExecutionEngine,
        dialect="mssql",
        partial_fn_type=MetricPartialFunctionTypes.WINDOW_CONDITION_FN,
    )
    def _sqlalchemy_window(cls, column, _table, **kwargs):
        dup_query = (
            sa.select([column])
            .select_from(_table)
            .group_by(column)
            .having(sa.func.count(column) > 1)
        )

        return column.notin_(dup_query)

    @column_condition_partial(
        engine=SparkDFExecutionEngine,
        partial_fn_type=MetricPartialFunctionTypes.WINDOW_CONDITION_FN,
    )
    def _spark(cls, column, **kwargs):
        return F.count(F.lit(1)).over(Window.partitionBy(column)) <= 1

    @classmethod
    def _get_evaluation_dependencies(
        cls,
        metric: MetricConfiguration,
        configuration: Optional[ExpectationConfiguration] = None,
        execution_engine: Optional[ExecutionEngine] = None,
        runtime_configuration: Optional[dict] = None,
    ):
        dependencies = super()._get_evaluation_dependencies(
            metric, configuration, execution_engine, runtime_configuration
        )
        metric_name = metric.metric_name
        if (
            metric_name.endswith(".unexpected_count")
            and execution_engine.dialect == "mssql"
        ):
            # we mustn't use aggregate_fn in this case, since we'll be using a window_fn
            # use the metric_value_kwargs computed for the partial_fn though
            configuration = dependencies.pop("metric_partial_fn")
            dependencies["unexpected_condition"] = MetricConfiguration(
                metric_name[: -len(".unexpected_count")] + ".condition",
                metric.metric_domain_kwargs,
                configuration.metric_value_kwargs,
            )
        return dependencies
