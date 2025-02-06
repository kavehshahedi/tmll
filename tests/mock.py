import uuid
from dataclasses import dataclass
from typing import List, Optional, Dict, Any
from unittest.mock import Mock
import numpy as np

from tmll.common.models.experiment import Experiment
from tmll.common.models.output import Output
from tmll.common.models.tree.node import NodeTree
from tmll.common.models.tree.tree import Tree
from tmll.common.models.trace import Trace
from tmll.tsp.tsp.indexing_status import IndexingStatus


@dataclass
class MockConfig:
    """Default configuration for mock objects"""
    TRACE_NAME: str = "test-trace"
    TRACE_PATH: str = "tmll/tests/test-trace/kernel"
    EXPERIMENT_NAME: str = "test-experiment"
    START_TIME: int = 1735689600000  # 2025-01-01 00:00:00
    END_TIME: int = 1735693200000    # 2025-01-01 01:00:00
    NUMBER_OF_EVENTS: int = 1000
    SERIES_ID: str = "test.series.id"
    SERIES_NAME: str = "Test Series"
    OUTPUT_ID: str = "test.output.id"
    OUTPUT_NAME: str = "Test Output"
    OUTPUT_TYPE: str = "TABLE"
    DEFAULT_INDEX_STATUS: IndexingStatus = IndexingStatus.COMPLETED
    STATUS_CODE: int = 200
    STATUS_TEXT: str = "OK"


class TMLLMockData:
    """
    Centralized mock data provider for TMLL tests.
    Provides mock objects and data structures commonly used in TMLL testing.
    """

    @staticmethod
    def _generate_uuid(identifier: str) -> uuid.UUID:
        """
        Generate a consistent UUID for a given identifier

        :param identifier: The identifier to generate the UUID from
        :type identifier: str
        :return: The generated UUID
        :rtype: uuid.UUID
        """
        return uuid.uuid5(uuid.NAMESPACE_DNS, identifier)

    @staticmethod
    def _create_base_mock(spec: Any, **kwargs) -> Mock:
        """
        Create a base mock object with common attributes

        :param spec: The spec of the mock object
        :type spec: Any
        :param kwargs: The keyword arguments to set as attributes
        :type kwargs: Dict[str, Any]
        :return: The created mock object
        :rtype: Mock
        """
        mock_obj = Mock(spec=spec)
        for key, value in kwargs.items():
            setattr(mock_obj, key, value)
        return mock_obj

    @classmethod
    def create_mock_trace(cls, name: str = MockConfig.TRACE_NAME, path: str = MockConfig.TRACE_PATH,
                          start: int = MockConfig.START_TIME, end: int = MockConfig.END_TIME, number_of_events: int = MockConfig.NUMBER_OF_EVENTS,
                          indexing_status: IndexingStatus = MockConfig.DEFAULT_INDEX_STATUS) -> Mock:
        """
        Create a mock Trace object with specified parameters.

        :param name: The name of the trace
        :type name: str
        :param path: The path of the trace
        :type path: str
        :param start: The start time of the trace
        :type start: int
        :param end: The end time of the trace
        :type end: int
        :param number_of_events: The number of events in the trace
        :type number_of_events: int
        :param indexing_status: The indexing status of the trace
        :type indexing_status: IndexingStatus
        :return: The created mock Trace object
        :rtype: Mock
        """
        return cls._create_base_mock(
            spec=Trace,
            UUID=cls._generate_uuid(f"{name}-{path}"),
            name=name,
            path=path,
            start=start,
            end=end,
            number_of_events=number_of_events,
            indexing_status=indexing_status
        )

    @classmethod
    def create_mock_experiment(cls, name: str = MockConfig.EXPERIMENT_NAME, traces: Optional[List[Mock]] = None,
                               start_time: int = MockConfig.START_TIME, end_time: int = MockConfig.END_TIME, number_of_events: int = MockConfig.NUMBER_OF_EVENTS,
                               indexing_status: IndexingStatus = MockConfig.DEFAULT_INDEX_STATUS, outputs: Optional[List[Output]] = None) -> Mock:
        """
        Create a mock Experiment object with specified parameters.

        :param name: The name of the experiment
        :type name: str
        :param traces: The traces of the experiment
        :type traces: Optional[List[Mock]]
        :param start_time: The start time of the experiment
        :type start_time: int
        :param end_time: The end time of the experiment
        :type end_time: int
        :param number_of_events: The number of events in the experiment
        :type number_of_events: int
        :param indexing_status: The indexing status of the experiment
        :type indexing_status: IndexingStatus
        :param outputs: The outputs of the experiment
        :type outputs: Optional[List[Output]]
        :return: The created mock Experiment object
        :rtype: Mock
        """
        experiment_uuid = cls._generate_uuid(name)
        traces_mock = Mock()
        traces_mock.traces = traces or []

        return cls._create_base_mock(
            spec=Experiment,
            UUID=experiment_uuid,
            uuid=experiment_uuid,
            name=name,
            traces=traces_mock,
            start=start_time,
            end=end_time,
            number_of_events=number_of_events,
            outputs=outputs or [],
            indexing_status=indexing_status
        )

    @staticmethod
    def create_mock_tree(nodes: List[Dict[str, Any]]) -> Mock:
        """
        Create a mock Tree object with specified node structure.

        :param nodes: The nodes of the tree with id, labels, and parent_id
        :type nodes: List[Dict[str, Any]]
        :return: The created mock Tree object
        :rtype: Mock
        """
        tree = Mock(spec=Tree)

        # Create mock nodes with NodeTree structure
        mock_nodes = [
            Mock(
                spec=NodeTree,
                id=node["id"],
                labels=node["labels"],
                parent_id=node.get("parent_id")
            ) for node in nodes
        ]
        tree.entries = mock_nodes

        def get_node_by_id(node_id: int) -> Optional[Mock]:
            return next((node for node in mock_nodes if node.id == node_id), None)

        tree.get_node_by_id = get_node_by_id
        tree.get_node_parent = lambda node_id: (
            get_node_by_id(node.parent_id) if (node := get_node_by_id(node_id))
            and hasattr(node, 'parent_id') and node.parent_id is not None
            else None
        )

        return tree

    @classmethod
    def create_mock_output(cls, output_id: str = MockConfig.OUTPUT_ID, name: str = MockConfig.OUTPUT_NAME,
                           output_type: str = MockConfig.OUTPUT_TYPE) -> Mock:
        """
        Create a mock Output object with specified parameters.

        :param output_id: The ID of the output
        :type output_id: str
        :param name: The name of the output
        :type name: str
        :param output_type: The type of the output
        :type output_type: str
        :return: The created mock Output object
        :rtype: Mock
        """
        return cls._create_base_mock(
            spec=Output,
            id=output_id,
            name=name,
            type=output_type
        )

    @classmethod
    def create_mock_xy_data(cls, start_time: int = MockConfig.START_TIME, end_time: int = MockConfig.END_TIME,
                            number_of_events: int = MockConfig.NUMBER_OF_EVENTS,
                            series_name: str = MockConfig.SERIES_NAME, series_id: str = MockConfig.SERIES_ID) -> Mock:
        """
        Create mock XY data with time series.

        :param start_time: The start time of the time series
        :type start_time: int
        :param end_time: The end time of the time series
        :type end_time: int
        :param number_of_events: The number of events in the time series
        :type number_of_events: int
        :param series_id: The ID of the series
        :type series_id: str
        :param series_name: The name of the series
        :type series_name: str
        :return: The created mock XY data object
        :rtype: Mock
        """
        x_values = np.linspace(start_time, end_time, number_of_events, dtype=int).tolist()
        y_values = np.random.randint(0, 100, number_of_events).tolist()

        series = [Mock(
            series_id=series_id,
            series_name=series_name,
            x_values=x_values,
            y_values=y_values
        )]

        return cls._create_base_mock(Mock, series=series)

    @staticmethod
    def create_mock_table_columns(num_columns: int = 3, column_names: Optional[List[str]] = None) -> List[Mock]:
        """
        Create mock table columns with specified parameters.

        :param num_columns: The number of columns to create
        :type num_columns: int
        :param column_names: The names of the columns
        :type column_names: Optional[List[str]]
        :return: The created mock table columns
        :rtype: List[Mock]
        """
        if column_names:
            columns = [Mock(id=str(i)) for i in range(1, len(column_names) + 1)]
            for i, column in enumerate(columns, 1):
                column.name = column_names[i - 1]
        else:
            columns = [Mock(id=str(i)) for i in range(1, num_columns + 1)]
            for i, column in enumerate(columns, 1):
                column.name = f"C{i}"

        return columns

    @classmethod
    def create_mock_table_data(cls, columns: List[Mock], rows_data: List[List[str]], low_index: int = 0) -> Mock:
        """
        Create mock table data with specified rows.

        :param columns: The columns of the table
        :type columns: List[Mock]
        :param rows_data: The data of the rows
        :type rows_data: List[List[str]]
        :param low_index: The low index of the table
        :type low_index: int
        :return: The created mock table data object
        :rtype: Mock
        """
        mock_lines = [
            Mock(
                index=low_index + idx,
                cells=[Mock(content=value) for value in row_values]
            )
            for idx, row_values in enumerate(rows_data)
        ]

        return cls._create_base_mock(
            Mock,
            columns=columns,
            size=len(rows_data),
            low_index=low_index,
            column_ids=[col.id for col in columns],
            lines=mock_lines
        )

    @classmethod
    def create_mock_timegraph_states(cls, states_data: List[Dict[str, Any]]) -> Mock:
        """
        Create mock timegraph states data.

        :param states_data: The states data with entry_id, states, start_time, end_time, and label
        :type states_data: List[Dict[str, Any]]
        :return: The created mock timegraph states object
        :rtype: Mock
        """
        mock_rows = [
            Mock(
                entry_id=entry_data["entry_id"],
                states=[
                    Mock(
                        start_time=state["start_time"],
                        end_time=state["end_time"],
                        label=state["label"]
                    )
                    for state in entry_data["states"]
                ]
            )
            for entry_data in states_data
        ]

        return cls._create_base_mock(Mock, rows=mock_rows)

    @classmethod
    def create_mock_tsp_response(cls, status_code: int = MockConfig.STATUS_CODE, model: Optional[Mock] = None,
                                 status_text: str = MockConfig.STATUS_TEXT) -> Mock:
        """
        Create a mock TSP response with specified parameters.

        :param status_code: The status code of the response
        :type status_code: int
        :param model: The model of the response
        :type model: Optional[Mock]
        :param status_text: The status text of the response
        :type status_text: str
        :return: The created mock TSP response object
        :rtype: Mock
        """
        return cls._create_base_mock(
            Mock,
            status_code=status_code,
            model=model,
            status_text=status_text
        )
