# Ayakchi Google Earth Engine environmental data

Open environmental inputs downloaded for the Climate Information and Early Warning System. See `manifest.json` for dataset IDs, periods, resolution, units and provenance. Climate rasters retain their native information content even when exported in UTM; CHIRPS (~5.6 km) and ERA5-Land (~11.1 km) must not be interpreted as field-scale spatial data. Daily CSVs are basin-average series.

Reproduce with `python download_gee_environmental_data.py --project ee-sabitovty`.
