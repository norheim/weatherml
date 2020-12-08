using DataFrames, CSV, Impute, Dates, Plots, Statistics, Interpolations
include("utils.jl");

choice = "boston"

df = CSV.File(string("../", choice, ".csv")) |> DataFrame

# Extract daily and hourly time series
dfdaily = filter(row -> strip(row[:REPORT_TYPE]) == "SOD", df)
# these are the entries that correspond to daily and monthly data which we exclude;
dfh = filter(row -> !(strip(row[:REPORT_TYPE]) in ["SOD", "SOM"]), df)

factors =  [
    (:HourlyDryBulbTemperature, :temp), 
    (:HourlyPrecipitation, :prec), 
    (:HourlySeaLevelPressure, :pres)
    ]

mdata = selectdata(dfh, factors);


