# From Python to Elixir: A Machine Learning Case Study

In the world of machine learning, Python is often considered the go-to language for developing and deploying models. However, Elixir - a functional, concurrent programming language built on top of the Erlang virtual machine - is also well-suited for machine learning tasks. In this blog post, we will explore a real-world machine learning problem in both Python and Elixir

We will be using data from the NSW Water API to analyse rainfall and stream height measurements surrounding Warragamba Dam, a large dam located in New South Wales, Australia. The goal is to try and predict change in height of the dam for the next day, given the weather of the current day.

### Imports & Setup

For this project I implemented the Python using [Jupyter notebook](https://jupyter.org/) and the Elixir with [Livebook](https://livebook.dev/).

Livebook is Elixir’s code notebook offering, it runs on top of [Phoenix](https://www.phoenixframework.org/). It allows you to program solutions interactively, collaboratively and saves to live markdown so its easy to share.  

In this project Python dependencies are installed using `Poetry` . This is the relevant portion of the `pyproject.toml`  and the import at the top of my Jupyter notebook:

```python
# pyproject.toml
[tool.poetry.dependencies]
python = ">=3.8,<3.12"
pandas = "^1.5.2"
numpy = "^1.24.0"
tensorflow = "^2.11.0"
requests = "^2.28.1"
altair-saver = "^0.5.0"

[tool.poetry.group.dev.dependencies]
altair = "^4.2.0"
jupyter = "^1.0.0"

# top of Jupyter notebook
import numpy as np # Numerical python lib
import pandas as pd # Data frame lib
import altair as alt # Chart plotting lib
import tensorflow as tf # Neural Network lib
import requests
import json
import pprint

from tensorflow.keras import layers
```

In the Elixir Livebook its a similar story:

```elixir
Mix.install(
  [
    {:httpoison, "~> 1.8"}, # HTTP client
    {:jason, "~> 1.4"}, # JSON encoder/decoder
    {:vega_lite, "~> 0.1.5"}, # Elixir Vega-lite binding
    {:kino, "~> 0.8.0"}, # Provides beautiful outputs for Livebook
    {:kino_vega_lite, "~> 0.1.1"}, # Kino utils for Vega-lite
    {:explorer, "~> 0.4.0"}, # Data frame library build on Polars
    {:axon, "~> 0.3.0"}, # Neural network library compiles to exla, torchlib through nx
    {:exla, "~> 0.4.0"}, # Elixir's binding for Google's XLA Linear Algebra Optimiser
    {:nx, "~> 0.4.0"}, # Elixir's numerical computation library
 ],
  config: [
    nx: [default_backend: EXLA.Backend]
  ]
)

# Sets the global compilation options
Nx.Defn.global_default_options(compiler: EXLA)
# Sets the process-level compilation options
Nx.Defn.default_options(compiler: EXLA)

alias VegaLite, as: Vl
alias Explorer.DataFrame, as: DF
alias Explorer.Series
```

## Data Extraction

First we need to download the data from its source and put it into a container of some kind. In Python this would be a `[pandas` DataFrame](https://pandas.pydata.org/) and in Elixir it would be an `Explorer` DataFrame. `[Explorer` is a DataFrame for Elixir](https://github.com/elixir-nx/explorer) which uses `Polars` under the hood, a library written in rust widely accepted as the fastest DataFrame out there.

I’ll gloss over the [details of the API itself](https://github.com/andrewcowley/WaterNSW-data-API-documentation). All you need to know is it returns time series data in a JSON format where `v` is the data **value**, `t` is the **time** the measurement was taken and `q` is the **quality code** for the measurement of the item. Its in this format:

```jsx
[{"t": "20230101000000", "v": "-0.124", "q":"5"}, ...]
```

To get the data from the API in Python:

```python
# see here for more information about stations https://realtimedata.waternsw.com.au/
weather_station_ids = ["563035", "563046", "563079", "568045", "568051"]
stream_station_ids = ["212250", "212270"]

def cast_data_types(df):
    df["v"] = df["v"].astype(float)
    df["q"] = df["q"].astype(float)
    df["t"] = pd.to_datetime(df["t"], format="%Y%m%d%H%M%S")
    return df
    
json_response = get_ts_trace(...)
water_level_df = pd.json_normalize(json_response)
cast_data_types(water_level_df)

rainfall_dfs = []
for station in weather_station_ids:
    json_response = get_ts_trace(station, ...)
    
    rainfall_dfs.append(
        cast_data_types(
            pd.json_normalize(json_response)
        )
    )

stream_dfs = []
for station in stream_station_ids:
    json_response = get_ts_trace(station, ...)
    
    stream_dfs.append(
        cast_data_types(
            pd.json_normalize(json_response)
        )
    )
```

To get the data from the API in Elixir:

```elixir
# see here for more information about stations https://realtimedata.waternsw.com.au/
weather_station_ids = ["563035", "563046", "563079", "568045", "568051"]
stream_station_ids = ["212250", "212270"]

water_level_df =
  WaterAPI.get_ts_trace(...)
  |> DF.new()
  |> DF.mutate(v: Series.cast(v, :float))

rainfall_dfs =
  for station_id <- weather_station_ids  do
     WaterAPI.get_ts_trace(station_id, ...)
    |> DF.new()
    |> DF.mutate(v: Series.cast(v, :float))
  end

stream_dfs =
  for station_id <- stream_station_ids  do
     WaterAPI.get_ts_trace(station_id, ...)
    |> DF.new()
    |> DF.mutate(v: Series.cast(v, :float))
  end
```

We do this for the **rainfall and stream data too**, which means we end up with **8 pieces** of time series data in total:

- **1 measuring** Warragamba Dam **water level** stored in `water_level_df`
- **2 measuring the major rivers** which flow into the dam, **stored as a list of DataFrames** in `stream_dfs`
- **5 measuring the rainfall** at weather stations near the dam, **stored at a list of DataFrames** in `rainfall_dfs`

## Deriving ‘Water Level Difference’

The data we are receiving is the average water level of the dam each day. From this we need to derive the **change in water level** each day, as this is what we want our neural network model to predict.

This can be done like so in Python:

```python
water_level = water_level_df["v"]
water_level_tomorrow = water_level_df["v"].copy().shift(1, fill_value=0.0)
water_level_df["water_level_difference"] = water_level - water_level_tomorrow
```

Elixir:

```elixir
water_level_tomorrow =
  water_level_df["v"]
  |> Series.shift(-1)
  |> then(fn s -> DF.new(water_level_tomorrow: s) end)

water_level_df =
  water_level_df
  |> DF.concat_columns(water_level_tomorrow)
  |> DF.mutate(water_level_difference: water_level_tomorrow - v)
  |> DF.discard("water_level_tomorrow")
```

Notice how here we have to `.copy()` the a column in Python to prevent `.shift()` from causing a side effect on column `"v"` when calculating the `"water_level_difference"`. This is something you have to consider fairly often when using Pandas, some functions give you views into the data, others give you copies of the data. In Elixir and with Explorer its easy, everything is immutable all of the time.

## Cleaning

We need to make sure there are no invalid values in our `water_level_df` as it will negatively effect our predictions. A value `v` is invalid when the corresponding quality value `q` is `201` or `255`.

For our `stream_dfs` and `rainfall_dfs` we will set invalid values to 0**.** This is ok because the neural network should ‘figure out’ that this value means ‘no input’ and treat it accordingly.

Python:

```python
water_level_df = water_level_df[~water_level_df["q"].isin([201,205])]

for stream_df in stream_dfs:
    stream_df.loc[stream_df["q"].isin([201,255]), "v"] = 0.0
    
for rainfall_df in rainfall_dfs:
    rainfall_df.loc[rainfall_df["q"].isin([201,255]), "v"] = 0.0
```

Elixir:

```elixir
water_level_df =
  DF.filter(
    water_level_df,
    q != 201 and q != 255
)

rainfall_dfs =
  for df <- rainfall_dfs do
    df
    |> DF.mutate(m: Series.cast(q != 201 and q != 255, :integer))
    |> DF.mutate(v: v * m)
    |> DF.discard(:m)
  end

stream_dfs =
  for df <- stream_dfs do
    df
    |> DF.mutate(m: Series.cast(q != 201 and q != 255, :integer))
    |> DF.mutate(v: v * m)
    |> DF.discard(:m)
  end
```

**Notice how the Python code is difficult to understand** without knowing how Pandas works. The Elixir code is much more self explanatory.

However the **Elixir code could be more succinct.** There should really be a dedicated way for what is being done here.

## Renaming and Joining

We want to join this data together into one DataFrame. However each DataFrame has the same 3 column names `t,v,q` we want to join on the time column `t`, so we need to rename the `v` and `q` columns to something unique for each data frame.

But what unique thing should we change the column name to? I decided to rename them by appending the `station_id` and the `variable` number to the column name.

Finally we need to join all of the data frames together with an **[inner join](https://www.w3resource.com/sql/joins/perform-an-inner-join.php).** This is important because **we want to ensure that only rows are included where there is corresponding data across all the DataFrames (for a given timestamp `t` ).**

Renaming and joining the data frames in Python:

```python
def rename_cols_with_id(dfs, station_ids: list, variable_no: str):
    for df, station_id in zip(dfs, station_ids):
        df.rename(
            columns={
                "v": f"v_{station_id}_{variable_no}",
                "q": f"q_{station_id}_{variable_no}",
            },
            inplace=True,
        )

water_level_df = water_level_df.rename(
    columns={"v": "v_212242_130", "q": "q_212242_130"}
)

rename_cols_with_id(rainfall_dfs, weather_station_ids, "10")
rename_cols_with_id(stream_dfs, stream_station_ids, "100")

df = water_level_df

for rainfall_df in rainfall_dfs:
    df = pd.merge(left=df, right=rainfall_df, how="inner", on="t")

for stream_df in stream_dfs:
    df = pd.merge(left=df, right=stream_df, how="inner", on="t")

df.columns
# ['v_212242_130', 't', 'q_212242_130', 'water_level_difference',
#  'v_563035_10', 'q_563035_10', 'v_563046_10', 'q_563046_10',
#  'v_563079_10', 'q_563079_10', 'v_568045_10', 'q_568045_10',
#  'v_568051_10', 'q_568051_10', 'v_212250_100', 'q_212250_100',
#  'v_212270_100', 'q_212270_100']
```

Elixir:

```elixir
defmodule Helper do
  def rename_cols_by_id(dfs, station_ids, variable_no) do
    for {df, station_id} <- Enum.zip(dfs, station_ids) do
      DF.rename(
        df,
        q: "q_#{station_id}_#{variable_no}",
        v: "v_#{station_id}_#{variable_no}"
      )
    end
  end

  def join_dfs(dfs) when is_list(dfs) do
    Enum.reduce(dfs, fn df, acc ->
      DF.join(df, acc, how: :inner)
    end)
  end

	...
end

water_level_df =
	DF.rename(water_level_df, q: "q_212242_130", v: "v_212242_130")

rainfall_df =
  rainfall_dfs
  |> Helper.rename_cols_by_id(weather_station_ids, "10")
	|> Helper.join_dfs()

stream_df =
	stream_dfs
	|> Helper.rename_cols_by_id(stream_station_ids, "100")
	|> Helper.join_dfs()

df =
	water_level_df
	|> DF.join(rainfall_df)
	|> DF.join(stream_df)

DF.names(df)
# ["q_212242_130", "t", "v_212242_130", "water_level_difference", "q_568051_10", "v_568051_10",
#  "q_568045_10", "v_568045_10", "q_563079_10", "v_563079_10", "q_563046_10", "v_563046_10",
#  "q_563035_10", "v_563035_10", "q_212270_100", "v_212270_100", "q_212250_100", "v_212250_100"]

```

Our data is now cleaned and in a single place, lets step back to look at what we’ve achieved. We’ve:

- Set up our journal and imported our dependency.
- Downloaded 8 pieces of time series data (via an API) and place each of them in a data frame.
- Created a new column which measures the change in water level of the dam
- Removed all invalid values from the table
- Renamed columns of all the data frames to something meaningful and unique
- Joined all the data frames together into one single data frame

## Prepare Data for the Model

We have our data neatly in one data frame. Now we need to sort it into:

- Training data
- Testing data
- Validation data (just in the Elixir Implementation)

Each of these categories are subdivided further into:

- Features (input to the model)
- Labels (expected input to the model)

We want 80% of our data for training and 20% for testing. The training data will be split further so that 20% of the training data is used for validation while training. The features will be the rainfall and stream level data. The labels will be the water of the dam.

You’ll notice that here we’ll be using the Elixir library `Nx` . `Nx` is a multi-dimensional tensors library for Elixir. The real beauty of it is that the tensors can target 3 backends, Elixir Native code, Google XLA (same backend as TensorFlow) and LibTorch. Enabling you neural networks to have similar performance to PyTorch and TensorFlow. [See our introductory post about Nx.](https://alembic.com.au/blog/high-performance-numerical-elixir-with-nx)

Here’s the Python code:

```python
feature_columns = [
  "v_568051_10", "v_568045_10", "v_563079_10",
  "v_563046_10", "v_563035_10", "v_212250_100",
  "v_212270_100"]

label_column = "water_level_difference"
model_columns = feature_columns + [label_column]

train_data = df.copy()[model_columns].sample(frac=0.8, random_state=12345)
test_data = df.copy()[model_columns].drop(train_data.index)

train_features = train_data.copy()
test_features = test_data.copy()

train_labels = train_features.pop("water_level_difference").values.reshape(-1, 1)
test_labels = test_features.pop("water_level_difference").values.reshape(-1, 1)
```

And the Elixir code:

```elixir

defmodule Helper do

	...

	def split_data(df, decimal) do
    {first, second} =
      df
      |> DF.to_rows()
      |> Enum.shuffle()
      |> split_by_decimal(decimal)

    {DF.new(first), DF.new(second)}
  end

  def df_to_tensor(df) do
    df
    |> DF.names()
    |> Enum.map(&Explorer.Series.to_tensor(df[&1]))
    |> Nx.stack(axis: 1)
  end

  def df_to_tensor_batches(df, batch_size) do
    df
    |> df_to_tensor()
    |> Nx.shuffle(axis: 0)
    |> Nx.to_batched(batch_size, leftover: :discard)
  end

  defp split_by_decimal(enum, decimal) do
    row_no = Enum.count(enum)
    Enum.split(enum, round(decimal * row_no))
  end
end

feature_columns = [
  "v_568051_10",
  "v_568045_10",
  "v_563079_10",
  "v_563046_10",
  "v_563035_10",
  "v_212250_100",
  "v_212270_100"
]

label_column = "water_level_difference"
model_columns = [label_column | feature_columns]

df = DF.select(df, model_columns)

# normalising data
df = Helper.normalise_df(df)

{train_df, test_df} = Helper.split_data(df, 0.8)
{train_df, validation_df} = Helper.split_data(train_df, 0.8)

train_features = Helper.df_to_tensor_batches(train_df[feature_columns], 8)
train_label = Helper.df_to_tensor_batches(train_df[[label_column]], 8)

validation_features = Helper.df_to_tensor_batches(validation_df[feature_columns], 1)
validation_label = Helper.df_to_tensor_batches(validation_df[[label_column]], 1)

test_features = Helper.df_to_tensor_batches(test_df[feature_columns], 1)
test_label = Helper.df_to_tensor_batches(test_df[[label_column]], 1)

training_batches =
  [train_features, train_label]
  |> Stream.zip()

validation_batches =
  [validation_features, validation_label]
  |> Stream.zip()

testing_batches =
  [test_features, test_label]
  |> Stream.zip()
```

As you can see the Elixir code is a little more involved. `Axon` (Elixir’s Neural Network library) is choosey about the format it receives the data, it won’t take an `Explorer` DataFrame, only  `Nx` zipped batches of  `Nx` tensors. However there isn’t an obvious, optimal way to convert an `Explorer` data frame into an `Nx` tensor. The same is true for splitting the data, there is no ‘out of the box’ way of doing it.

## Building the Model

Now we have all our data in the right format we are ready to build our neural network. Funnily enough this is actually the easiest part. In out python version we will use `TensorFlow` and in Elixir we’re going to use `Axon`.

`[Axon` is a neural network library built completely on top of `Nx`](https://github.com/elixir-nx/axon). `Axon` has a bunch of sensible APIs that are simple enough for a beginner but flexible enough that an expert can easily do *just* what they want.

Lets take a look at the TensorFlow model in Python first, then compare with Elixir.

Python:

```python
normaliser = tf.keras.layers.Normalization(axis=1)
normaliser.adapt(train_features)

test_model = tf.keras.Sequential(name="stream_and_rain_model", layers=[
    layers.Input(shape=(7,)),
    normaliser,
    layers.Dropout(rate=0.5),
    layers.Dense(units=16, activation="relu"),
    layers.Dropout(rate=0.5),
    layers.Dense(units=1)
])
```

Elixir:

```elixir
model =
  Axon.input("stream_and_rain_model")
  |> Axon.dropout(rate: 0.5)
  |> Axon.dense(16)
	|> Axon.relu()
  |> Axon.dropout(rate: 0.5)
  |> Axon.dense(1)
```

You can see how these are both remarkably similar, the only major difference is that the python model has a normalising layer. There isn’t a layer for this in Axon yet, as its also simple to do the transformation yourself before the features are input to the neural network. As was done in the previous Elixir example.

## Training

Python:

```python
test_model.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
    loss='mean_absolute_error')

history = test_model.fit(
    train_features.values,
    train_labels,
    epochs=30,
    # Suppress logging.
    verbose=1,
    # Calculate validation results on 20% of the training data.
    validation_split = 0.2)
```

Elixir:

```elixir
model_params =
  model
  |> Axon.Loop.trainer(:mean_absolute_error, Axon.Optimizers.adam(0.001))
  |> Axon.Loop.validate(model, validation_batches)
  |> Axon.Loop.metric(:mean_absolute_error, "validation_loss")
  |> Axon.Loop.run(training_batches, %{}, epochs: 30)

```

The semantics are almost exactly the same, we just see differences in the API. We see that Python does a little more for you here, remember how we didn’t need to make a `validation_batches` variable? We see thats because we can simply tell Python to validate while training for us.

## Results

We can see the model is training by observing the training history. While you do get this information each epoch in `Axon` sadly, I couldn’t figure out a way to extract it.

Here is the Python code:

```python
hist_df = pd.DataFrame(history.history)
hist_df["epoch"] = history.epoch
hist_df.rename(
    columns={"loss":"training_loss", "val_loss":"validation_loss"},
    inplace=True
)

alt.Chart(hist_df).mark_line().transform_fold(
    fold=['training_loss', 'validation_loss'], 
    as_=['variable', 'loss']
).encode(
    x="epoch:Q",
    y="loss:Q",
    color="variable:N"
)
```

It produces:

![Untitled](From%20Python%20to%20Elixir%20A%20Machine%20Learning%20Case%20Stud%2057b3db3c10764aad947b93dffe652ce2/Untitled.svg)

We can clearly see the training loss and the validation loss going down.

We can use our test data to evaluate our model too.

Python:

```python
test_model.evaluate(test_features, test_labels)
# 29/29 [==============================] - 0s 1ms/step - loss: 0.0208
```

Elixir:

```elixir
model
|> Axon.Loop.evaluator()
|> Axon.Loop.metric(:mean_absolute_error)
|> Axon.Loop.run(testing_batches, model_params, epoch: 1)
# Batch: 927, mean_absolute_error: 0.0245324
```

This loss values of 0.0208 and 0.0245 mean that on average the model is within approximately 0.02m of the real change in dam water level.

We can observe this visually by feeding the model all of the feature data, and then comparing the predictions of the change in water level with the real change in water level.

Here is the Python code for that:

```python
y = test_model.predict(df[feature_columns])

compare_df = pd.DataFrame({
    "t": df[["t"]].values.flatten(),
    "actual": df[["water_level_difference"]].values.flatten(),
    "prediction": y.flatten()
})

base = alt.Chart(compare_df.reset_index()[0:300]).encode(
    x="index:Q"
)

(base.mark_line().encode(
    y="actual:Q"
) + base.mark_line(color="orange").encode(
    y="prediction:Q"
)).interactive()
```

And the Elixir code:

```elixir
all_features = df_to_tensor.(df[feature_columns])

all_labels =
  df[label_column]
  |> Series.to_tensor()
  |> Nx.to_flat_list()

{_init_fn, predict_fn} = Axon.build(model, mode: :inference)

predictions =
  predict_fn.(model_params, all_features)
  |> Nx.to_flat_list()

row_no = 4640

chart_data =
  DataFrame.new(
    prediction: predictions,
    actual: all_labels,
    count: Enum.map(1..row_no, & &1)
  )
  |> DataFrame.to_rows()

Vl.new(width: 400, height: 400)
|> Vl.data_from_values(Enum.take(chart_data, 300))
|> Vl.layers([
  Vl.new()
  |> Vl.param("prediction_chart", select: :interval, bind: :scales, encodings: ["x", "y"])
  |> Vl.encode(:y, field: "prediction", type: :quantitative)
  |> Vl.encode(:x, field: "count", type: :quantitative)
  |> Vl.mark(:line, color: "orange"),
  Vl.new()
  |> Vl.encode(:x, field: "count", type: :quantitative)
  |> Vl.encode(:y, field: "actual", type: :quantitative)
  |> Vl.mark(:line, color: "blue")
])
```

This produces:

![Untitled](From%20Python%20to%20Elixir%20A%20Machine%20Learning%20Case%20Stud%2057b3db3c10764aad947b93dffe652ce2/Untitled%201.svg)

The orange line is model’s prediction, the blue line is the ground truth. We can see that it seems to predict when there is a spike quite well, but not necessarily its magnitude. The model seems to have a slight systematic bias towards predicting lower values than reality. I think this may be because we don’t have any data about the outflow of the dam.

I think for a model where the only data its getting is from the day before and it has to predict the next day, its performing remarkably well. This model could be improved of course with more data sources but also but inputting multiple days worth of data at once. The purpose of this post was to compare Python and Elixir with a real problem, so thats outside the scope of this post. If I’ve piqued your interest you can see both notebooks [here](https://github.com/team-alembic/warragamba-blogpost) (which also includes a multi day model in python with much better accuracy).

## Conclusion

**Elixir is punching way above its weight**

I was not expecting Elixir to be better at Python for data science, and its not. The python data science ecosystem is huge and had a vast amount of money poured into it by big tech.

However its important to acknowledge just how far Elixir’s data science has some in the past 2 years. Its gone from 0 to being a viable option for a data science project. Its done this by standing on the shoulders of giants, and taken all the best bits from the whole ecosystem, and packaged them together in a simple but flexible way.

**The Elixir data science ecosystem is new and you can feel it.**

Have a problem? The chances are you won’t be able to find a blog post about it, or a StackOverflow answer, you’ll just have to scour the docs for it and hope it exists. This is already pain point in Elixir but it’s even more apparent with these new data science libraries. 

I found this tended to occur at the boundary from one library to another. For example, why can’t I transform my `Explorer` DataFrame into a 2D `Nx` tensor? Or why can’t I input a `Explorer` Series into `Axon` and then it apply the conversion to a tensor for me, we have pattern matching don’t we? This is compounded by the fact that there are no examples besides the docs of how to do things properly or efficiently, then you have to improvise your own, probably inefficient, solution.

Its not a deal breaker and I’m sure it will improve but it breaks your flow to find you have to build tools which should be there.

**It feels better to program in Elixir**

Call me biased (because I might be), but it felt way better programming in Elixir. I found the code easier to reason about and more understandable.

The main reason for this is that the Elixir was way more readable, just look at this example:

```elixir
# python
water_level_df = water_level_df[~water_level_df["q"].isin([201,205])]

# elixir
water_level_df =
  DF.filter(
    water_level_df,
    q != 201 and q != 255
)
```

The Elixir code is 4 lines and the python code is a one-liner. But whats more important is that a Junior developer would know what it was doing without having to look at Stack Overflow.

I also found that I was never looking over my shoulder, as I was in Python, considering if any of the functions I’d just used returned a reference to an object, or whether the data was copied then modified. I found this especially with `pandas` and it actually caused a couple of pesky bugs, which encourages you to defensively `.copy()` your data which isn’t good for performance.

Contrast this with Elixir. I knew that I was always passing an immutable object. It removed a whole category of errors from your project.

When I was programming in Python I really missed my pipes in Elixir. I found myself making intermediate variables where I would have otherwise used an Elixir pipe. Its called a ‘data pipeline' isn’t it? So why isn't there an elegant way to pipe in Python?

All in all this was a rewarding experience, I’d still say for now python is the Boring™️ but optimal option. From the outset you know it can do whatever you want to do, they’ll be a way to do it, and if you get stuck they’ll be a plethora of docs and posts about that exact error. That’s just not a guarantee you get with Elixir data science tooling right now.

However the Elixir data science tooling is still very young, 2 years ago it was practically non-existent and now its viable and performant option if you want to wrangle some data or train a model. Python had a 10 year head start but if it carries on at this rate I may be telling you Elixir is the clear winner in 2 years time.