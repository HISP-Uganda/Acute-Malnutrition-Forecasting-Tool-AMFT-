# Sample Data Folder

The app loads these files by default:

- `data.csv`: 8,760 monthly rows for 146 districts, spanning January 2021 to December 2025.
- `district_map.geojson`: boundaries for 146 districts. Use the `name` field to match district labels in the CSV.

The CSV includes under-five malaria, pneumonia, diarrhoea, screening, and population covariates, along with climate data, low birth weight, reporting rate, and GAM caseload (`Acut_Malnutrition`). See the [main README](../../README.md#data-requirements) for the column schema.

There are 555 missing GAM caseload values, leaving 8,205 observations with reported caseloads. These are input counts, not the number of usable training rows or eligible forecast districts.

Missing caseloads are excluded from analysis, not filled with zero. Training also excludes feature rows with missing covariates. Some districts therefore have too little usable history for forecasting. The under-five population remains a distinct covariate and is not treated as total population.

The common forecast cutoff is December 2025. Only districts with an observed caseload at that cutoff and usable training history receive January–March 2026 forecasts. Districts without a December observation are excluded and shown as `No Data` on forecast maps.

Run the application from the project root with `python -m streamlit run app.py`. The input paths can be changed in the sidebar, or files can be uploaded instead.

Do not commit sensitive health data. The full working dataset should stay in `data/private/`, which is ignored by Git.
