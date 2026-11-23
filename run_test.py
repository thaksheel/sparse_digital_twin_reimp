from dataclasses import replace

from dataset import OilGprDataset
from pipeline import DependenceDiscovery
from results import ResultExporter

QUICK_TEST = True

main_periods = [
    OilGprDataset(name="oil_gpr_1986_2026"),
    OilGprDataset(name="oil_gpr_1986_1999", rows=(0, 3550)),
    OilGprDataset(name="oil_gpr_2000_2009", rows=(3550, 6056)),
    OilGprDataset(name="oil_gpr_2010_2026", rows=(6056, None)),
]

five_year_periods = [
    OilGprDataset(name="oil_gpr_1987_1991", rows=(251, 1532)),
    OilGprDataset(name="oil_gpr_1992_1996", rows=(1532, 2796)),
    OilGprDataset(name="oil_gpr_1997_2001", rows=(2796, 4050)),
    OilGprDataset(name="oil_gpr_2002_2006", rows=(4050, 5299)),
    OilGprDataset(name="oil_gpr_2007_2011", rows=(5299, 6560)),
    OilGprDataset(name="oil_gpr_2012_2016", rows=(6560, 7820)),
    OilGprDataset(name="oil_gpr_2017_2021", rows=(7820, 9071)),
    OilGprDataset(name="oil_gpr_2022_2026", rows=(9071, 10142)),
]

if QUICK_TEST:
    main_periods = [replace(d, epochs=50, n_distill=1000, eval_samples=50) for d in main_periods[:1]]
    five_year_periods = []

exporter = ResultExporter("results")

for group, datasets in [("main_periods", main_periods), ("five_year_periods", five_year_periods)]:
    results = []
    for dataset in datasets:
        result = DependenceDiscovery(dataset).run()
        exporter.export(result)
        results.append(result)
        print(f"{result.name}: last pruned = {result.last_pruned}, selected = {result.selected}")
    if results:
        exporter.export_comparison(results, group)
