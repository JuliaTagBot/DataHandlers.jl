__precompile__(true)

# TODO remove preprocessing stuff, it doesn't belong here
# TODO implement padding for TimeSeriesHandler

module DataHandlers

using Reexport

@reexport using DataTables
using DataUtils

include("DataHandler.jl")
include("TimeSeriesHandler.jl")
include("GroupedDataHandler.jl")










end
