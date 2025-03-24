from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional, Tuple

import matplotlib
import matplotlib.lines
import matplotlib.pyplot as plt
from matplotlib.figure import Figure
from matplotlib.axes import Axes

from segretini_matplottini.plot import roofline  # type: ignore


@dataclass
class CPUSpecification:
    name: str
    cores: int
    threads: int
    base_clock: float  # GHz
    boost_clock: float  # GHz

    max_memory_bandwidth: float  # GB/s

    l1_cache: float  # KB
    l2_cache: float  # KB
    l3_cache: float  # MB

    theoretical_flops: float = -1  # GFLOPS, calculated if not provided
    operational_intensity: float = 0.5  # FLOP/Byte, default example value
    measured_performance: float = -1  # GFLOPS, can be measured or estimated

    def __post_init__(self):
        if self.theoretical_flops < 0:
            ops_per_cycle = 32  # AVX-512 support
            self.theoretical_flops = self.cores * self.boost_clock * ops_per_cycle


@dataclass
class DataPoint:
    name: str
    operational_intensity: float  # FLOP/Byte
    measured_performance: float   # GFLOPS
    marker: str = 'o'


class CPURooflineModel:

    def __init__(self, plot_dir: Path = Path("./plots")):
        """Initialize the CPURooflineModel.

        :param plot_dir: Directory to save plots
        :type plot_dir: Path
        """
        self.plot_dir = plot_dir
        self.plot_dir.mkdir(exist_ok=True, parents=True)

        self.markers = ['o', 's', '^', 'D', 'v', 'x', '*', '+', 'P', 'H']
        self.palette = [plt.get_cmap('tab10')(i) for i in range(10)]

    def create_roofline_plot(self, cpus: List[CPUSpecification],
                             data_points: Optional[List[DataPoint]] = None,
                             xmin: float = 0.1,
                             xmax: float = 20,
                             ymin: Optional[float] = None,
                             ymax: Optional[float] = None,
                             figure_size: Tuple[float, float] = (10, 8),
                             title: str = "CPU Roofline Model Comparison") -> Tuple[Figure, Axes]:
        """
        Create a roofline plot for multiple CPUs.

        :param cpus: List of CPU specifications
        :type cpus: List[CPUSpecification]
        :param data_points: List of data points to plot
        :type data_points: Optional[List[DataPoint]]
        :param xmin: Minimum x-axis value
        :type xmin: float
        :param xmax: Maximum x-axis value
        :type xmax: float
        :param ymin: Minimum y-axis value
        :type ymin: Optional[float]
        :param ymax: Maximum y-axis value
        :type ymax: Optional[float]
        :param figure_size: Size of the figure
        :type figure_size: Tuple[float, float]
        :param title: Title of the plot
        :type title: str
        :return: Figure and Axes objects
        :rtype: Tuple[Figure, Axes]
        """
        performance = []
        operational_intensity = []
        peak_performance = []
        peak_bandwidth = []
        labels = []

        for cpu in cpus:
            # Convert bandwidth from GB/s to B/s
            bw_bytes_per_sec = cpu.max_memory_bandwidth * 1e9

            # Convert theoretical FLOPS from GFLOPS to FLOPS
            peak_flops = cpu.theoretical_flops * 1e9

            peak_performance.append(peak_flops)
            peak_bandwidth.append(bw_bytes_per_sec)

            operational_intensity.append(cpu.operational_intensity)

            if cpu.measured_performance is not None:
                perf = cpu.measured_performance * 1e9
            else:
                perf = peak_flops * 0.7
            performance.append(perf)

            labels.append(cpu.name)

        fig, ax = roofline(
            performance=performance,
            operational_intensity=operational_intensity,
            peak_performance=peak_performance,
            peak_bandwidth=peak_bandwidth,
            palette=self.palette[:len(cpus)],  # type: ignore
            markers=self.markers[:len(cpus)],
            performance_unit="FLOPS",
            xmin=xmin,
            xmax=xmax,
            add_legend=True,
            legend_labels=labels,
            figure_size=figure_size
        )

        ax.set_title(title)

        cpu_handles = [matplotlib.lines.Line2D(
            [0], [0], color=self.palette[i % len(self.palette)], label=label
        ) for i, label in enumerate(labels)]

        cpu_legend = ax.legend(handles=cpu_handles, loc='upper left', title="CPUs")
        ax.add_artist(cpu_legend)

        if ymin is not None and ymax is not None:
            ax.set_ylim(ymin, ymax)

        if data_points:
            for dp in data_points:
                # Convert GFLOPS to FLOPS
                perf = dp.measured_performance * 1e9
                ax.loglog(
                    dp.operational_intensity,
                    perf,
                    marker=dp.marker,
                    markersize=8,
                    linestyle='none',
                    alpha=0.7
                )

            unique_workloads = set([dp.name for dp in data_points])
            dp_legend = []
            for i, workload in enumerate(unique_workloads):
                dp_legend.append(matplotlib.lines.Line2D(
                    [0], [0],
                    marker=self.markers[i % len(self.markers)],
                    color=self.palette[i % len(self.palette)],
                    markersize=8,
                    linestyle='none',
                    label=workload
                ))

            ax.legend(handles=dp_legend, loc='lower right', title="Data Points")

        return fig, ax

    def save_plot(self, fig: Figure, plot_name: str = "cpu_roofline") -> Path:
        """
        Save the plot to a file.

        :param fig: Figure object to save
        :type fig: Figure
        :param plot_name: Name of the plot file
        :type plot_name: str
        :return: Path to the saved plot file
        :rtype: Path
        """
        file_path = self.plot_dir / f"{plot_name}.png"
        fig.savefig(file_path, dpi=300, bbox_inches='tight')
        return file_path
