from typing import Dict, List, Optional, Union

from great_expectations.core.expectation_configuration import ExpectationConfiguration

from ..expectation import ColumnMapExpectation, InvalidExpectationConfigurationError
from ..metrics import ColumnValuesZScore


class ExpectColumnValueEntropyToBeLessThan(ColumnMapExpectation):
    # Todo: fix documentation after materializing implementation
    """
    Expect the Z-scores of a columns values to be less than a given threshold

            expect_column_values_to_be_of_type is a :func:`column_map_expectation \
            <great_expectations.execution_engine.execution_engine.MetaExecutionEngine.column_map_expectation>` for
            typed-column
            backends,
            and also for PandasExecutionEngine where the column dtype and provided type_ are unambiguous constraints (any
            dtype
            except 'object' or dtype of 'object' with type_ specified as 'object').

            Parameters:
                column (str): \
                    The column name of a numerical column.
                threshold (number): \
                    A maximum entropy threshold in the interval [0,100], which can be interpreted as the Likelihood of anomaly.
                    All column entropy values that are lower than this threshold (less anomalous than it) will evaluate
                    successfully.



            Keyword Args:
                mostly (None or a float between 0 and 1): \
                    Return `"success": True` if at least mostly fraction of values match the expectation. \
                    For more detail, see :ref:`mostly`.
                method (string): \
                    A string value indicating the method to be used in the estimation of entropy. Potential values are the
                    following:

                    "bootstrap" - using a training set, the model will build an expectation for categorical value counts
                    and estimate entropy via divergence from this Expectation. Values that are found far more or less
                    often in the dataset than Expected would be considered anomalous.

                    "kde" - Using a Kernel Density approximation, the model will consider the least dense areas of a density
                    plot (locations where there are very few or no values in the vicinity) most anomalous. Can also provide
                    a training set and base anomaly on densities that are most different between the train and test set.

                    "categorical" - learning the presence of certain syntax patterns within each categorical value
                    within the column, the model will build Expectations for which rules the data follows more or less
                    often. The more and stricter the rules that a data point breaks, the more anomalous and "entropic" it
                    will be considered.




            Other Parameters:
                result_format (str or None): \
                    Which output mode to use: `BOOLEAN_ONLY`, `BASIC`, `COMPLETE`, or `SUMMARY`.
                    For more detail, see :ref:`result_format <result_format>`.
                include_config (boolean): \
                    If True, then include the Expectation config as part of the result object. \
                    For more detail, see :ref:`include_config`.
                catch_exceptions (boolean or None): \
                    If True, then catch exceptions and include them as part of the result object. \
                    For more detail, see :ref:`catch_exceptions`.
                meta (dict or None): \
                    A JSON-serializable dictionary (nesting allowed) that will be included in the output without \
                    modification. For more detail, see :ref:`meta`.

            Returns:
                An ExpectationSuiteValidationResult

                Exact fields vary depending on the values passed to :ref:`result_format <result_format>` and
                :ref:`include_config`, :ref:`catch_exceptions`, and :ref:`meta`.
    """

    # Setting necessary computation metric dependencies and defining kwargs, as well as assigning kwargs default values\
    map_metric = "column_values.entropy.under_threshold"
    success_keys = ("threshold", "method", "double_sided", "mostly")

    # Default values
    default_kwarg_values = {
        "row_condition": None,
        "condition_parser": None,
        "threshold": None,
        "double_sided": True,
        "method": "kde",
        "mostly": 1,
        "result_format": "BASIC",
        "include_config": True,
        "catch_exceptions": False,
    }

    def validate_configuration(self, configuration: Optional[ExpectationConfiguration]):
        """
        Validates that a configuration has been set, and sets a configuration if it has yet to be set. Ensures that
        neccessary configuration arguments have been provided for the validation of the expectation.

        Args:
            configuration (OPTIONAL[ExpectationConfiguration]): \
                An optional Expectation Configuration entry that will be used to configure the expectation
        Returns:
            True if the configuration has been validated successfully. Otherwise, raises an exception
        """

        # Setting up a configuration
        super().validate_configuration(configuration)
        if configuration is None:
            configuration = self.configuration
        try:
            # Ensuring entropy threshold properly provided
            assert (
                "threshold" in configuration.kwargs
            ), "A threshold must be provided"
            assert isinstance(
                configuration.kwargs["threshold"], (float, int, dict)
            ), "Provided threshold must be a number"
            if isinstance(configuration.kwargs["threshold"], dict):
                assert (
                    "$PARAMETER" in configuration.kwargs["threshold"]
                ), 'Evaluation Parameter dict for threshold kwarg must have "$PARAMETER" key.'

            assert "method" not in configuration.kwargs or isinstance(
                configuration.kwargs["method"], (bool, dict)
            ), "Provided method argument must be a string"
        except AssertionError as e:
            raise InvalidExpectationConfigurationError(str(e))
        return True
