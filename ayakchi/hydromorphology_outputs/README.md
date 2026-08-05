# Ayakchi hydromorphology — phase 2

The verified `Ayakchi Dam (Option -5)` point in the supplied KMZ is used as the outlet. All valid DEM cells are routed to it using an outlet-seeded priority-flood D8 tree. The conditioned DEM should be treated as an analytical routing surface, not observed terrain.

The drainage network uses a 1 km² contributing-area threshold. GeoTIFFs are GIS-ready; GeoJSON files contain the outlet, basin boundary, main flow path, and stream segments. CSVs contain the basin metrics, hypsometric curve, and longitudinal profile.
