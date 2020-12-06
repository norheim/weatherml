function clean(x)
    if ismissing(x) || x isa Float64
        return x
    end
    m = match(r"[+-]?([0-9]*[.])?[0-9]+",x)
    return m ==nothing ? missing : parse(Float64, m.match)
end

function resample(dfh, factor)
    dfh[!,factor] = clean.(dfh[:,factor])
    timeseries = Impute.interp(dfh[!, [:DATE, factor]]) |> Impute.locf() |> Impute.nocb() 
    timeseries[!, factor] = convert.(Float64, timeseries[!, factor]) # remove the missing type
    # even sampling
    timeseries[!, :rounded] = map((d) -> round(d, Dates.Hour), timeseries[!, :DATE])
    even_sampling = by(timeseries, :rounded, factor => mean)
    rename!(even_sampling, Symbol(factor, "_mean") => :data)
    return even_sampling
end

function selectdata(dfh, factors)
    factor, frename = first(factors)
    mdata = resample(dfh, factor)
    rename!(mdata, Dict(:data => frename, :rounded => :sampleT));
    for (factor, frename) in factors[2:end]
        mdata[:, frename] = resample(dfh, factor)[:, :data]
    end
    return mdata
end

monthdayhour = x -> [Dates.month(x) Dates.day(x) Dates.hour(x)];

function build_multi_data(mdata, predicttype, start, dayslookback, ndays; offset=1, hour=0)
    _, ntypes = size(mdata) .- 1
    year, month, startday = start
    additional_factors = 3                    # for month, day, hour info
    historical_factors = 24*dayslookback
    ntype_factors = ntypes*historical_factors
    total_factors = ntype_factors + additional_factors
    n_datapoints = 24*ndays
    X = zeros(n_datapoints, total_factors)
    y = zeros(n_datapoints);
    t0 = DateTime(year, month, startday, hour, 0)
    dateidx = findall(x -> x== t0, mdata[:, :sampleT])[1];
    hist = reduce(vcat, (mdata[dateidx:dateidx+historical_factors-1,2:end] |> Matrix)')'
    for idx=1:n_datapoints
        next_point = dateidx+historical_factors+idx-1
        predict = next_point+offset-1
        predict_date = mdata[predict,:sampleT]
        X[idx, 1:total_factors] = [hist monthdayhour(predict_date)]
        y[idx] = mdata[predict,predicttype]
        hist = [hist[(1+ntypes):end]' (mdata[next_point, 2:end] |> Vector)']
    end
   return X,y
end