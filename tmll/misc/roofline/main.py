from pathlib import Path
from tmll.misc.roofline.roofline import CPURooflineModel, CPUSpecification, DataPoint


def main():
    model = CPURooflineModel(plot_dir=Path("./output"))

    cpus = [
        CPUSpecification(
            name="AMD Ryzen 9 9950X3D",
            cores=16,
            threads=32,
            base_clock=4.3,
            boost_clock=5.7,
            max_memory_bandwidth=44.8,  # GB/s
            l1_cache=80,  # KB
            l2_cache=1024,  # KB
            l3_cache=128,    # MB
            operational_intensity=2,  # FLOP/Byte
        ),
        CPUSpecification(
            name="Intel Core i5-14400",
            cores=10,
            threads=16,
            base_clock=2.5,
            boost_clock=4.7,
            max_memory_bandwidth=38.4,  # GB/s
            l1_cache=80,  # KB
            l2_cache=1.25*1024,  # KB
            l3_cache=20,    # MB
            operational_intensity=0.5,  # FLOP/Byte
        ),
        CPUSpecification(
            name="Qualcomm Snapdragon X1P-26-100",
            cores=8,
            threads=8,
            base_clock=3.0,
            boost_clock=3.0,
            max_memory_bandwidth=67.5,  # GB/s
            l1_cache=288,  # KB
            l2_cache=12*1024,  # KB
            l3_cache=6,    # MB
            operational_intensity=0.5,  # FLOP/Byte
        )
    ]

    data_points = [
        DataPoint(
            name="DP1",
            operational_intensity=500,  # FLOP/Byte
            measured_performance=100.0,  # GFLOPS
            marker="o",
        ),
        DataPoint(
            name="DP2",
            operational_intensity=1000,  # FLOP/Byte
            measured_performance=200.0,  # GFLOPS
            marker="s",
        ),
    ]

    fig, _ = model.create_roofline_plot(
        cpus=cpus,
        data_points=data_points,
        title="CPU Roofline Comparison",
        xmax=(max(dp.operational_intensity for dp in data_points) * 3)
    )

    model.save_plot(fig, "custom_cpu_roofline")


if __name__ == "__main__":
    main()
