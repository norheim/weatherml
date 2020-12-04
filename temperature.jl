using DataFrames, CSV, Tables, Dates, Missings, Impute, Plots,
    Interpolations, Statistics, LinearAlgebra

df = CSV.File("weather_cleaned_v1.csv",
    types=Dict(:HourlyDryBulbTemperature=>Float64)) |> DataFrame

# Extract daily and hourly time series
dfdaily = filter(row -> strip(row[:REPORT_TYPE]) == "SOD", df)
# these are the entries that correspond to daily and monthly data which we exclude;
dfh = filter(row -> !(strip(row[:REPORT_TYPE]) in ["SOD", "SOM"]), df)

# Temperature data
temp = Impute.interp(dfh[!, [:DATE,:HourlyDryBulbTemperature]]);
temp[!, :HourlyDryBulbTemperature] = convert.(Float64, temp[!, :HourlyDryBulbTemperature]);

# Sample evenly
temp[!, :rounded] = map((d) -> round(d, Dates.Hour), temp[!, :DATE]);
temp_even_sampling = by(temp, :rounded, :HourlyDryBulbTemperature => mean);

# adding month/day/hour
historical_data = 5 # number of days for which we have historical data(at hourly intervals)
t0 = DateTime(2018, 1, 1+historical_data, 0, 0)
training_days = 360
X = zeros(24*training_days, 24*historical_data + 3)
y = zeros(24*training_days)
for day=1:training_days
    for hour=0:23
        date = t0 + Day(day) + Hour(hour)
        X[24*(day-1)+hour+1, 1:(24*historical_data)] = filter(row -> date - Day(historical_data)<= row[:rounded] <= date - Hour(1), temp_even_sampling)[!,:HourlyDryBulbTemperature_mean]
        X[24*(day-1)+hour+1, 24*historical_data + 1] = Dates.month(date);
        X[24*(day-1)+hour+1, 24*historical_data + 2] = Dates.day(date);
        X[24*(day-1)+hour+1, 24*historical_data + 3] = Dates.hour(date);
        y[24*(day-1)+hour+1] = first(filter(row -> row[:rounded]==date, temp_even_sampling))[:HourlyDryBulbTemperature_mean]
    end
end

(train_X, train_y), (test_X, test_y) = IAI.split_data(:regression, X, y);
grid = IAI.GridSearch(
    IAI.OptimalTreeRegressor(
        random_seed=123,
        show_progress=false
        minbucket=10
    ),
    max_depth=5:10,
    show_progress=false # <-- uncomment to avoid having all the progress bars show up
)
IAI.fit!(grid, train_X, train_y)
lnr = IAI.get_learner(grid)
IAI.score(lnr, test_X, test_y)
