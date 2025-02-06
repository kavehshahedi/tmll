from typing import List
import pandas as pd
import pytest
from unittest.mock import Mock, patch
from tmll.common.models.tree.tree import Tree
from tmll.tmll_client import TMLLClient
from tests.mock import TMLLMockData, MockConfig
from tmll.tsp.tsp.indexing_status import IndexingStatus

@pytest.fixture
def mock_data():
    """Fixture to provide access to mock data generator."""
    return TMLLMockData()

class TestTMLLClient:
    """Test suite for the TMLLClient class."""

    @pytest.fixture(autouse=True)
    def setup(self):
        """Set up the test environment with mocked TSP client."""
        with patch("tmll.tmll_client.TspClient") as tsp_patcher:
            self.mock_tsp = tsp_patcher.return_value
            yield

    def _init_client(self, mock_data: TMLLMockData, status_code: int = 200) -> TMLLClient:
        """Helper method to initialize client with mock health response."""
        self.mock_tsp.fetch_health.return_value = mock_data.create_mock_tsp_response(status_code=status_code)
        return TMLLClient()

    def _setup_trace_mocks(self, mock_data: TMLLMockData, traces: List[Mock]):
        """Helper method to set up trace-related mock responses."""
        self.mock_tsp.open_trace.side_effect = [
            mock_data.create_mock_tsp_response(status_code=200, model=trace)
            for trace in traces
        ]

    def _setup_experiment_mocks(self, mock_data: TMLLMockData, mock_experiment: Mock):
        """Helper method to set up experiment-related mock responses."""
        self.mock_tsp.open_experiment.return_value = mock_data.create_mock_tsp_response(
            status_code=200,
            model=mock_experiment
        )
        self.mock_tsp.fetch_experiment.return_value = mock_data.create_mock_tsp_response(
            status_code=200,
            model=mock_experiment
        )
        self.mock_tsp.fetch_experiment_outputs.return_value = mock_data.create_mock_tsp_response(
            status_code=200,
            model=Mock(descriptors=[])
        )

    @pytest.mark.parametrize("status_code,expected_status", [(200, 200), (400, 400)])
    def test_client_initialization(self, mock_data: TMLLMockData, status_code: int, expected_status: int):
        """Test client initialization with various health status codes."""
        client = self._init_client(mock_data, status_code)
        assert client.health_status == expected_status
        self.mock_tsp.fetch_health.assert_called_once()

    def test_create_experiment_successful(self, mock_data: TMLLMockData):
        """Test successful experiment creation with valid traces."""
        mock_traces = [
            mock_data.create_mock_trace(name=f"trace{i}", path=f"/path/to/trace{i}")
            for i in range(1, 3)
        ]
        mock_experiment = mock_data.create_mock_experiment(traces=mock_traces)
        
        self._setup_trace_mocks(mock_data, mock_traces)
        self._setup_experiment_mocks(mock_data, mock_experiment)
        
        client = self._init_client(mock_data)
        traces = [{"name": t.name, "path": t.path, "UUID": t.UUID} for t in mock_traces]
        experiment = client.create_experiment(traces, mock_experiment.name)

        # Verify experiment creation
        assert experiment is not None
        assert experiment.name == mock_experiment.name
        assert experiment.start == mock_experiment.start
        assert experiment.end == mock_experiment.end
        assert experiment.num_events == mock_experiment.number_of_events
        assert len(experiment.traces) == len(mock_traces)
        assert experiment.outputs == []

        # Verify API calls
        assert self.mock_tsp.open_trace.call_count == len(mock_traces)
        self.mock_tsp.open_experiment.assert_called_once()
        self.mock_tsp.fetch_experiment.assert_called_once()
        self.mock_tsp.fetch_experiment_outputs.assert_called_once()

    def test_create_experiment_corner_cases(self, mock_data: TMLLMockData):
        """Test experiment creation corner cases."""
        client = self._init_client(mock_data)

        # Test with empty traces
        assert client.create_experiment([], MockConfig.EXPERIMENT_NAME) is None
        self.mock_tsp.open_trace.assert_not_called()

        # Test with failed trace opening
        self.mock_tsp.open_trace.return_value = mock_data.create_mock_tsp_response(
            status_code=400,
            status_text="Failed to open trace"
        )
        traces = [{"path": f"/path/to/trace{i}"} for i in range(1, 3)]
        assert client.create_experiment(traces, MockConfig.EXPERIMENT_NAME) is None

    def test_fetch_outputs_with_tree(self, mock_data: TMLLMockData):
        """Test fetching outputs with tree data."""
        mock_experiment = mock_data.create_mock_experiment()
        mock_output = mock_data.create_mock_output(output_type="TABLE")
        mock_experiment.outputs = [mock_output]

        tree_data = [
            {"id": "1", "labels": ["Root"]},
            {"id": "2", "labels": ["Child1"], "parent_id": "1"},
            {"id": "3", "labels": ["Child2"], "parent_id": "1"}
        ]
        mock_tree = mock_data.create_mock_tree(tree_data)

        self.mock_tsp.fetch_datatree.return_value = mock_data.create_mock_tsp_response(
            status_code=200,
            model=Mock(status=IndexingStatus.COMPLETED, model=mock_tree)
        )

        client = self._init_client(mock_data)
        result = client.fetch_outputs_with_tree(mock_experiment)

        assert result is not None
        assert isinstance(result, list)
        assert len(result) == 1
        assert result[0]["output"] == mock_output
        assert result[0]["tree"] == Tree.from_tsp_tree(mock_tree)
        self.mock_tsp.fetch_datatree.assert_called_once()

    def test_fetch_data_by_output_type(self, mock_data: TMLLMockData):
        """Test fetching data for different output types."""
        test_cases = {
            "TREE_TIME_XY": self._test_xy_output,
            "TABLE": self._test_table_output,
            "TIME_GRAPH": self._test_timegraph_output
        }

        for output_type, test_func in test_cases.items():
            mock_experiment = mock_data.create_mock_experiment()
            mock_output = mock_data.create_mock_output(output_type=output_type)
            mock_tree = self._create_basic_tree(mock_data)
            
            self._init_client(mock_data)
            test_func(mock_data, mock_experiment, mock_output, mock_tree)

    def _create_basic_tree(self, mock_data: TMLLMockData):
        """A method to create a basic mock tree."""
        return mock_data.create_mock_tree([
            {"id": "1", "labels": ["Root"]},
            {"id": "2", "labels": ["Child1"], "parent_id": "1"},
            {"id": "3", "labels": ["Child2"], "parent_id": "1"}
        ])

    def _test_xy_output(self, mock_data: TMLLMockData, mock_experiment: Mock, mock_output: Mock, mock_tree: Mock):
        """Test XY output type data fetching."""
        mock_xy_data = mock_data.create_mock_xy_data(
            start_time=mock_experiment.start,
            end_time=mock_experiment.end,
            number_of_events=mock_experiment.number_of_events
        )

        self.mock_tsp.fetch_xy.return_value = mock_data.create_mock_tsp_response(
            status_code=200,
            model=Mock(model=mock_xy_data)
        )

        outputs = [{"output": mock_output, "tree": Tree.from_tsp_tree(mock_tree)}]
        result = self._init_client(mock_data).fetch_data(mock_experiment, outputs)

        assert result is not None
        assert isinstance(result, dict)
        assert mock_output.id in result
        
        df = result[mock_output.id][MockConfig.SERIES_NAME]
        assert isinstance(df, pd.DataFrame)
        assert len(df) == mock_experiment.number_of_events
        assert all(df["x"] == mock_xy_data.series[0].x_values)
        assert all(df["y"] == mock_xy_data.series[0].y_values)

    def _test_table_output(self, mock_data: TMLLMockData, mock_experiment: Mock, mock_output: Mock, mock_tree: Mock):
        """Test TABLE output type data fetching."""
        cols = ["A", "B", "C"]
        mock_columns = mock_data.create_mock_table_columns(column_names=cols)
        mock_table_data = [
            mock_data.create_mock_table_data(
                columns=mock_columns,
                rows_data=[["value1", "value2", "[key1=value3a, key2=value3b]"], ["value4", "value5", "value6"]],
                low_index=0
            ),
            mock_data.create_mock_table_data(columns=mock_columns, rows_data=[], low_index=2)
        ]

        self.mock_tsp.fetch_virtual_table_columns.return_value = mock_data.create_mock_tsp_response(
            status_code=200,
            model=Mock(model=Mock(columns=mock_columns))
        )
        self.mock_tsp.fetch_virtual_table_lines.side_effect = [
            mock_data.create_mock_tsp_response(status_code=200, model=Mock(model=data))
            for data in mock_table_data
        ]

        outputs = [{"output": mock_output, "tree": Tree.from_tsp_tree(mock_tree)}]
        result = self._init_client(mock_data).fetch_data(mock_experiment, outputs, separate_columns=True)

        assert result is not None
        assert isinstance(result, dict)
        assert mock_output.id in result

        df = result[mock_output.id]
        assert isinstance(df, pd.DataFrame)
        assert len(df) == 2
        assert (len(df.columns)) == len(mock_columns) + 2 # Additional columns for keys
        assert all(col in df.columns for col in cols + ["key1", "key2"])
        assert df.loc[[0], ["key1", "key2"]].notnull().all().all() # First row has values for key1 and key2
        assert df.loc[[1], ["key1", "key2"]].isnull().all().all() # Second row has no values for key1 and key2

    def _test_timegraph_output(self, mock_data: TMLLMockData, mock_experiment: Mock, mock_output: Mock, mock_tree: Mock):
        """Test TIME_GRAPH output type data fetching."""
        mock_timegraph_data = [
            mock_data.create_mock_timegraph_states([
                {
                    "entry_id": i,
                    "states": [
                        {"start_time": 0, "end_time": 100, "label": "X"},
                        {"start_time": 100, "end_time": 200, "label": "Y"}
                    ]
                } for i in range(2, 4)
            ]),
            mock_data.create_mock_timegraph_states([])
        ]

        self.mock_tsp.fetch_timegraph_states.side_effect = [
            mock_data.create_mock_tsp_response(status_code=200, model=Mock(model=data))
            for data in mock_timegraph_data
        ]

        outputs = [{"output": mock_output, "tree": Tree.from_tsp_tree(mock_tree)}]
        result = self._init_client(mock_data).fetch_data(mock_experiment, outputs)
        
        assert result is not None
        assert isinstance(result, dict)
        assert mock_output.id in result

        df = result[mock_output.id]
        assert isinstance(df, pd.DataFrame)
        assert len(df) == 4
        required_columns = ["entry_id", "entry_name", "parent_id", "start_time", "end_time", "label"]
        assert all(col in df.columns for col in required_columns)