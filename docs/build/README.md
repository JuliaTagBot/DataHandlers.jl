
<a id='DataHandlers.jl-1'></a>

# DataHandlers.jl


Tools for preprocessing and loading data for machine learning in pure Julia.


<a id='DataHandler-1'></a>

## DataHandler


The main functionality currently available in `DatasToolbox` is supplied with the  `DataHandler` type.  This type is designed for converting dataframes into usable machine learning data in the form of input, target pairs $X, y$.  The design philosophy is that hyper-parameter tuning and testing should be done completely separately.


<a id='DataHandler-Example-1'></a>

### DataHandler Example


As a somewhat silly example, suppose we have a dataframe which stores (experimentally  determined) particle properties, and we would like to classify the particle as a quark,  lepton, scalar, or gauge boson and that we are doing this with gradient boosted trees. Suppose the dataframe contains 4-momenta, estimated spin and electric charge of particles, all as floats.


The columns which we want to use as input for our classification can be, for example,


```julia
input_cols = [:E, :px, :py, :pz, :S, :Q]
```


(yes, this example is ridiculous for a number of reasons, but its just to demonstrate how things work).  From this we want to predict the type of particle so let's declare


```julia
output_cols = [:ptype]
```


Now we construct the `DataHandler` object.


```julia
dh = DataHandler{Float64}(df, input_cols=input_cols, output_cols=output_cols)
```


The type parameter is the type which the train and test data will ultimately be converted to.  For now it is assumed that these are all the same (this conversion to all the same type turns out to be useful for many machine learning methods).


Next, we can randomly split the dataframe into training and test (validation) sets.  


```julia
split!(dh, 0.2)
```


This assigns one fifth of the data to the test set, and the rest to the training set. In realistic scenarios, one more often has to do splits like


```julia
split!(dh, df[:Q] .> 0.0)
```


(or something less ridiculous, but you get the idea).  Once we have split the data we are  ready to extract it in useful form.  First call 


```julia
assign!(dh)
```


to convert the data to properly formatted arrays.  Then, to retrieve them do


```julia
X_train, y_train = getTrainData(dh, flatten=true)
```


The flatten argument ensures that `y_train` is a rank-1 array as opposed to a rank-2 array with 1 column.


At this point you can train your classifier however you normally would.  In this example we'll use the gradient boosted tree library `xgboost`


```julia
boost = xgboost(X_train, N_ITERATIONS, label=y_train, eta=η)
```


Then we can get the test data and perform a test


```julia
X_test, y_test = getTestData(dh, flatten=true)

ŷ = predict(boost, X_test)
```


There are also tools for analyzing the test data.  To create a useful dataframe for testing one can do


```julia
output = getTestAnalysisData(dh, ŷ, names=[:ptype_predicted])
```


The names argument specifies the name of the prediction column in the resulting dataframe. Alternatively one can just use `y_test` to create whatever test statistics or plots one wants.


<a id='TimeSeriesHandler-Example-1'></a>

### TimeSeriesHandler Example


A far more complicated data manipulation task is preparing time series data.  For this we supply the type `TimeSeriesHandler`.  Suppose we have a dataframe containing the columns `:y` and `:τ` where `:τ` is a time index.  To declare the handler


```julia
tsh = TimeSeriesHandler{Float64}(df, :τ, SEQ_LENGTH, input_cols=[:y], output_cols=[:y],
                                 normalize_cols=[:y])
```


Note that here the input and output columns are the same, because we are auto-regressing $y$ on itself, but the input and output columns can be whatever you want.  In this case we also want to center and rescale the data so that it is more appropriate as input to a recurrent neural network.  This option is also avaialbe for the `DataHandler` object and works the same way as it will in this example.  It is required that the input dataframe has a time index.  `SEQ_LENGTH` provides the length of sequences in the training and test data.


Now, one can create a train-test split and generate the properly formatted data.


```julia
split!(tsh, τ_split)
computeNormalizeParamters!(tsh)
normalize!(tsh)
assign!(tsh)

X_train, y_train = getTrainData(tsh)
```


The function `computeNormalizeParameters!` computes the parameters that are necessary for performing the centering and rescaling.  After `normalize!` is called the returned data will be properly normalized.  In this example `X_train` is a rank-3 tensor appropriate for input to recurrent neural networks.  To instead return a matrix where each row is of length `SEQ_LENGTH` one can call `getSquashedTrainData`.


`X_train, y_train` are properly formatted arrays that can be fed into the training function of whatever method you are using.


When testing a time series regression, it is often desirable to create a sequence of a  specified length by predicting on the previous $N$ points.  This requires extremely complicated data manipulation, but can be done with


```julia
ŷ = generateSequence(predict, tsh, PREDICTION_LENGTH)
```


The first argument should be the function that is used to make predictions.  This will generate a sequence by predicting on the last $N$ points of the training set, then  predicting on the last $N-1$ points of the training set and the 1 point which was just predicted, then the last $N-2$ points of the training set and the 2 points which were just predicted and so forth.  If the predicted sequence is of the same length as the test set one can still do


```julia
output = getTestAnalysisData(tsh, ŷ, names=[:ŷ])
```


<a id='API-Docs-1'></a>

## API Docs

<a id='DataHandlers.AbstractDH' href='#DataHandlers.AbstractDH'>#</a>
**`DataHandlers.AbstractDH`** &mdash; *Type*.



```
AbstractDH{T}
```

Abstract base class for data handler objects.

<a id='DataHandlers.DataHandler' href='#DataHandlers.DataHandler'>#</a>
**`DataHandlers.DataHandler`** &mdash; *Type*.



```
DataHandler{T} <: AbstractDH{T}
```

Type for handling datasets.  This is basically a wrapper for a dataframe with methods for splitting it into training and test sets and creating input and output numerical arrays.  It is intended that most reformatting of the dataframe is done before passing it to an instance of this type.

The parameter T specifies the datatype of the input, output arrays.

<a id='DataHandlers.TimeSeriesHandler' href='#DataHandlers.TimeSeriesHandler'>#</a>
**`DataHandlers.TimeSeriesHandler`** &mdash; *Type*.



```
TimeSeriesHandler{T} <: AbstractDH{T}
```

Type for handling time series data.  As with DataHandler it is intended taht most of the reformatting of the dataframe is done before passing it to an instance of this type.

The parameter T specifies the datatype of the input, output arrays.

<a id='DataHandlers.assign!-Tuple{DataHandlers.AbstractDH{T}}' href='#DataHandlers.assign!-Tuple{DataHandlers.AbstractDH{T}}'>#</a>
**`DataHandlers.assign!`** &mdash; *Method*.



```
assign!(dh::AbstractDH)
```

Assigns training and test data in the data handler.

<a id='DataHandlers.assign!-Tuple{DataHandlers.TimeSeriesHandler}' href='#DataHandlers.assign!-Tuple{DataHandlers.TimeSeriesHandler}'>#</a>
**`DataHandlers.assign!`** &mdash; *Method*.



```
assign!(dh::TimeSeriesHandler; sort::Bool=true)
```

Assigns both training and testing data for the `TimeSeriesHandler`.

<a id='DataHandlers.assignTest!-Tuple{DataHandlers.AbstractDH{T}}' href='#DataHandlers.assignTest!-Tuple{DataHandlers.AbstractDH{T}}'>#</a>
**`DataHandlers.assignTest!`** &mdash; *Method*.



```
assignTest!(dh::AbstractDH)
```

Assigns the test data in the data handler.

<a id='DataHandlers.assignTest!-Tuple{DataHandlers.TimeSeriesHandler}' href='#DataHandlers.assignTest!-Tuple{DataHandlers.TimeSeriesHandler}'>#</a>
**`DataHandlers.assignTest!`** &mdash; *Method*.



```
assignTest!(dh[, df; sort=true])
```

Assigns the test data.  X output will be of shape (samples, seq_length, seq_width). One should be extremely careful if not sorting.

If a dataframe is provided, data will be assigned from it.

Note that in the time series case this isn't very useful.  One should instead use one of the assigned prediction functions.

<a id='DataHandlers.assignTrain!-Tuple{DataHandlers.AbstractDH{T}}' href='#DataHandlers.assignTrain!-Tuple{DataHandlers.AbstractDH{T}}'>#</a>
**`DataHandlers.assignTrain!`** &mdash; *Method*.



```
assignTrain!(dh::AbstractDH)
```

Assigns the training data in the data handler so it can be retrieved in proper form.

<a id='DataHandlers.assignTrain!-Tuple{DataHandlers.TimeSeriesHandler}' href='#DataHandlers.assignTrain!-Tuple{DataHandlers.TimeSeriesHandler}'>#</a>
**`DataHandlers.assignTrain!`** &mdash; *Method*.



```
assignTrain!(dh[, df; sort=true, parallel=false])
```

Assigns the training data.  X output will be of shape (samples, seq_length, seq_width). If `sort` is true, will sort the dataframe first.  One should be extremely careful if `sort` is false.  If `parallel` is true the data will be generated in parallel (using  workers, not threads).  This is useful because this data manipulation is complicated and potentially slow.

If a dataframe is provided, data will be assigned from it.  Alternatively, one can provide a vector of dataframes.  This is useful because sequences which cross the boundaries of the dataframes will *not* be created.

**TODO** I'm pretty sure the parallel version isn't working right because it doesn't use shared arrays.  Revisit in v0.5 with threads.

<a id='DataHandlers.computeNormalizeParameters!-Tuple{DataHandlers.AbstractDH{T}}' href='#DataHandlers.computeNormalizeParameters!-Tuple{DataHandlers.AbstractDH{T}}'>#</a>
**`DataHandlers.computeNormalizeParameters!`** &mdash; *Method*.



```
computeNormalizeParameters!{T}(dh::AbstractDH{T}; dataset::Symbol=:dfTrain)
```

Gets the parameters for centering and rescaling from either the training dataset  (`dataset=:dfTrain`) or the test dataset (`dataset=:dfTest`).

Does this using the training dataframe by default, but can be set to use test. Exits normally if this doesn't need to be done for any columns.

This should always be called before `normalize!`, that way you have control over what dataset the parameters are computed from.

<a id='DataHandlers.generateSequence-Tuple{Function,DataHandlers.TimeSeriesHandler{T},Integer}' href='#DataHandlers.generateSequence-Tuple{Function,DataHandlers.TimeSeriesHandler{T},Integer}'>#</a>
**`DataHandlers.generateSequence`** &mdash; *Method*.



```
generateSequence(predict, dh, seq_length[, newcol_func; on_matrix=false])
```

Uses the supplied prediction function `predict` to generate a sequence of length `seq_length`. The sequence uses the end of the training dataset as initial input.

Note that this only makes sense when the output columns are a subset of the input columns.

If a function returning a dictionary or a dictionary of functions `newcol_func` is supplied, every time a new row of the input is generated, it will have columns specified by  `newcol_func`.  The dictionary should have keys equal to the column numbers of columns in the input matrix and values equal to functions that take a `Vector` (the previous input row) and output a new value for the column number given by the key.  The column numbers correspond to the index of the column in the specified input columns.

If `on_matrix` is true, the prediction function will take a matrix as input rather than a rank-3 tensor.

<a id='DataHandlers.generateTest-Tuple{Function,DataHandlers.TimeSeriesHandler}' href='#DataHandlers.generateTest-Tuple{Function,DataHandlers.TimeSeriesHandler}'>#</a>
**`DataHandlers.generateTest`** &mdash; *Method*.



```
generateTest(predict::Function, dh::TimeSeriesHandler; on_matrix::Bool=true)
```

Uses the supplied prediction function to attempt to predict the entire test set. Note that this assumes that the test set is ordered, sequential and immediately follows the training set.

See the documentation for `generateSequence`.

<a id='DataHandlers.getGroupedTestAnalysisData-Tuple{DataTables.DataTable,Array{Symbol,1},Array{Symbol,1}}' href='#DataHandlers.getGroupedTestAnalysisData-Tuple{DataTables.DataTable,Array{Symbol,1},Array{Symbol,1}}'>#</a>
**`DataHandlers.getGroupedTestAnalysisData`** &mdash; *Method*.



```
getGroupedTestAnalysisData(data, keycols[; names=[], squared_error=true])
getGroupedTestAnalysisData(dh, data, keycols[; names=[], squared_error=true])
getGroupedTestAnalysisData(gdh, data[; names=[], squared_error=true])
getGroupedTestAnalysisData(gdh, ŷ[; names=[], squared_error=true])
```

Groups the output of `getTestAnalysisData` by the columns `keycols`.  This is particularly useful for `GroupedDataHandler` where a typical use case is applying different estimators to different subsets of the data.  One can supply the output `getTestAnalysisData` as `data` or pass a `GroupedDataHandler` together with an output dictionary `ŷ`, in which case all the tables will be generated for you.

<a id='DataHandlers.getRawTestTarget-Tuple{DataHandlers.TimeSeriesHandler{T}}' href='#DataHandlers.getRawTestTarget-Tuple{DataHandlers.TimeSeriesHandler{T}}'>#</a>
**`DataHandlers.getRawTestTarget`** &mdash; *Method*.



```
getRawTestTarget(dh::TimeSeriesHandler)
```

Returns `y_test` directly from the dataframe for comparison with the output of generateTest.

<a id='DataHandlers.getSquashedTestMatrix-Tuple{DataHandlers.TimeSeriesHandler}' href='#DataHandlers.getSquashedTestMatrix-Tuple{DataHandlers.TimeSeriesHandler}'>#</a>
**`DataHandlers.getSquashedTestMatrix`** &mdash; *Method*.



```
getSquashedTestMatrix(dh::TimeSeriesHandler)
```

Gets a test input tensor in which all the inputs are arranged along a single axis (i.e. in a matrix).

Assumes the handler's X_test is defined.

<a id='DataHandlers.getSquashedTrainData-Tuple{DataHandlers.TimeSeriesHandler}' href='#DataHandlers.getSquashedTrainData-Tuple{DataHandlers.TimeSeriesHandler}'>#</a>
**`DataHandlers.getSquashedTrainData`** &mdash; *Method*.



```
getSquashedTrainData(dh::TimeSeriesHandler; flatten::Bool=false)
```

Gets the training X, y pair where X is squashed using `getSquahdedTrainMatrix`. If `flatten`, also flatten `y`.

<a id='DataHandlers.getSquashedTrainMatrix-Tuple{DataHandlers.TimeSeriesHandler}' href='#DataHandlers.getSquashedTrainMatrix-Tuple{DataHandlers.TimeSeriesHandler}'>#</a>
**`DataHandlers.getSquashedTrainMatrix`** &mdash; *Method*.



```
getSquashedTrainMatrix(dh::TimeSeriesHandler)
```

Gets a training input tensor in which all the inputs are arranged along a single axis (i.e. in a matrix).

Assumes the handler's X_train is defined.

<a id='DataHandlers.getTestAnalysisData-Tuple{DataHandlers.AbstractDH,Array}' href='#DataHandlers.getTestAnalysisData-Tuple{DataHandlers.AbstractDH,Array}'>#</a>
**`DataHandlers.getTestAnalysisData`** &mdash; *Method*.



**`DataHandler`**

```
getTestAnalysisData(dh::AbstractDH, ŷ::Array; names::Vector{Symbol}=Symbol[],
                    squared_error::Bool=true)
```

Creates a dataframe from the test dataframe and a supplied prediction.  

The array names supplies the names for the columns, otherwise will generate default names.

Also generates error columns which are the difference between predictions and test data. If `squared_error`, will also create a column with squared error.

Note that this currently does nothing to handle transformations of the data.

<a id='DataHandlers.getTestData-Tuple{DataHandlers.AbstractDH}' href='#DataHandlers.getTestData-Tuple{DataHandlers.AbstractDH}'>#</a>
**`DataHandlers.getTestData`** &mdash; *Method*.



```
getTestData(dh::AbstractDH; flatten::Bool=false)
```

Gets the test data input, output tuple `X, y`.

If `flatten`, attempts to flatten `y`.

<a id='DataHandlers.getTrainData-Tuple{DataHandlers.AbstractDH}' href='#DataHandlers.getTrainData-Tuple{DataHandlers.AbstractDH}'>#</a>
**`DataHandlers.getTrainData`** &mdash; *Method*.



```
getTrainData(dh::AbstractDH; flatten::Bool=false)
```

Gets the training data input, output tuple `X, y`.

If `flatten`, attempts to flatten `y`.

<a id='DataHandlers.normalize!-Tuple{DataHandlers.AbstractDH{T}}' href='#DataHandlers.normalize!-Tuple{DataHandlers.AbstractDH{T}}'>#</a>
**`DataHandlers.normalize!`** &mdash; *Method*.



```
normalize!{T}(dh::AbstractDH{T}; dataset::Symbol=:dfTrain)
normalizeTrain!(dh::AbstractDH)
normalizeTest!(dh::AbstractDH)
```

Centers and rescales the columns set by `normalize_cols` in the `DataHandler` constructor.

<a id='DataHandlers.shuffle!-Tuple{DataHandlers.AbstractDH}' href='#DataHandlers.shuffle!-Tuple{DataHandlers.AbstractDH}'>#</a>
**`DataHandlers.shuffle!`** &mdash; *Method*.



```
shuffle!(dh::AbstractDH)
```

Shuffles the main dataframe of the DataHandler.

<a id='DataHandlers.split!-Tuple{DataHandlers.AbstractDH,AbstractFloat}' href='#DataHandlers.split!-Tuple{DataHandlers.AbstractDH,AbstractFloat}'>#</a>
**`DataHandlers.split!`** &mdash; *Method*.



```
split!(dh::AbstractDH, testfrac::AbstractFloat; shuffle::Bool=false,
       assign::Bool=true)
```

Creates a train, test split by fraction.  The fraction given is the test fraction.

<a id='DataHandlers.split!-Tuple{DataHandlers.AbstractDH,BitArray}' href='#DataHandlers.split!-Tuple{DataHandlers.AbstractDH,BitArray}'>#</a>
**`DataHandlers.split!`** &mdash; *Method*.



```
split!(dh::AbstractDH, constraint::BitArray)
```

Splits the data into training and test sets using a BitArray that must correspond to elements of dh.df.  The elements of the dataframe for which the BitArray holds 1 will be in the test  set, the remaining elements will be in the training set.

<a id='DataHandlers.split!-Tuple{DataHandlers.AbstractDH,Integer}' href='#DataHandlers.split!-Tuple{DataHandlers.AbstractDH,Integer}'>#</a>
**`DataHandlers.split!`** &mdash; *Method*.



```
split!(dh::AbstractDH, index::Integer; assign::Bool=true)
```

Creates a train, test split by index.  The index given is the last index of the training set. If `assign`, this will assign the training and test data.

<a id='DataHandlers.split!-Tuple{DataHandlers.TimeSeriesHandler,Integer}' href='#DataHandlers.split!-Tuple{DataHandlers.TimeSeriesHandler,Integer}'>#</a>
**`DataHandlers.split!`** &mdash; *Method*.



```
split!(dh::TimeSeriesHandler, τ₀::Integer; assign::Bool=true, sort::Bool=true)
```

Splits the data by time-index.  All datapoints with τ up to and including the timeindex given (τ₀) will be in the training set, while all those with τ > τ₀ will be in  the test set.

<a id='DataHandlers.splitByNSequences!-Tuple{DataHandlers.TimeSeriesHandler,Integer}' href='#DataHandlers.splitByNSequences!-Tuple{DataHandlers.TimeSeriesHandler,Integer}'>#</a>
**`DataHandlers.splitByNSequences!`** &mdash; *Method*.



```
splitByNSequences!(dh::TimeSeriesHandler, n_sequences::Integer;
                   assign::Bool=true, sort::Bool=true)
```

Splits the dataframe by the number of sequences in the test set.  This does nothing to account for the possibility of missing data.

Note that the actual number of usable test sequences in the resulting test set is of course greater than n_sequences.

<a id='DataHandlers.trainOnMatrix-Tuple{Function,DataHandlers.TimeSeriesHandler}' href='#DataHandlers.trainOnMatrix-Tuple{Function,DataHandlers.TimeSeriesHandler}'>#</a>
**`DataHandlers.trainOnMatrix`** &mdash; *Method*.



```
trainOnMatrix(train::Function, dh::TimeSeriesHandler; flatten::Bool=true)
```

Trains an object designed to take a matrix (as opposed to a rank-3 tensor) as input. The first argument is the function used to train.

If flatten, the training output is converted to a vector.  Most methods that take matrix input take vector target data.

Assumes the function takes input of the form train(X, y).

<a id='DataHandlers.unnormalize!-Tuple{DataHandlers.AbstractDH{T},Array{T,2},Array{Symbol,1}}' href='#DataHandlers.unnormalize!-Tuple{DataHandlers.AbstractDH{T},Array{T,2},Array{Symbol,1}}'>#</a>
**`DataHandlers.unnormalize!`** &mdash; *Method*.



```
unnormalize!{T}(dh::AbstractDH{T}, X::Matrix{T}, cols::Vector{Symbol})
```

Performs the inverse of the centering and rescaling operations on a matrix. This can also be called on a single column with a `Symbol` as the last argument.

<a id='DataHandlers.@selectTest!-Tuple{Any,Any}' href='#DataHandlers.@selectTest!-Tuple{Any,Any}'>#</a>
**`DataHandlers.@selectTest!`** &mdash; *Macro*.



```
@selectTrain!(dh, expr)
```

Set the test set to be the subset of the `DataHandler`'s dataframe for which `expr` is true.  `expr` should be an expression which evaluates to a `Bool` with `Symbol`s corresponding to column names for values.  See the documentation for `@constrain` and `@split!`  for examples.

<a id='DataHandlers.@selectTrain!-Tuple{Any,Any}' href='#DataHandlers.@selectTrain!-Tuple{Any,Any}'>#</a>
**`DataHandlers.@selectTrain!`** &mdash; *Macro*.



```
@selectTrain!(dh, expr)
```

Set the training set to be the subset of the `DataHandler`'s dataframe for which `expr` is true.  `expr` should be an expression which evaluates to a `Bool` with `Symbol`s corresponding to column names for values.  See the documentation for `@constrain` and `@split!`  for examples.

<a id='DataHandlers.@split!-Tuple{Any,Any}' href='#DataHandlers.@split!-Tuple{Any,Any}'>#</a>
**`DataHandlers.@split!`** &mdash; *Macro*.



```
@split!(dh, expr)
```

Splits the `DataHandler`'s DataTable into training in test set, such that the test set is the set of datapoints for which `expr` is true, and the training set is the set of datapoints for which `expr` is false.  `expr` should be an expression that evaluates to `Bool` with symbols in place of column names.  For example, if the columns of the dataframe are `[:x1, :x2, :x3]` one can do `expr = (:x1 > 0.0) | (tanh(:x3) > e^6)`.

See documentation on the `@constrain` macro which `@split!` calls internally.

